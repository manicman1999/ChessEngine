import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from cychess import Board
from attn_model import AttnModelCompiled

class AttnModel(nn.Module):
    def __init__(self, num_tokens: int = 960, embed_dim: int = 24, head_dim_qk: int = 4, head_dim_v: int = 8):
        """
        Residual sliced attention for chess board eval.
        - Tokens: 14 piece variants × 64 squares = 896 ids (pos baked in).
        - Embed → [bs, 64, 24]: Q[0:4], K[4:8], V[8:16], E[16:24] (residual).
        - Attn on QKV → Z[64,8]; cat(Z, E) → [64,16] flat → MLP.
        - ~120k ops, ~38k params. Output: [bs] value logit.
        """
        super().__init__()
        assert embed_dim == (head_dim_qk * 2 + head_dim_v * 2)  # 4+4+8+8=24
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.head = nn.Linear(64 * (head_dim_v * 2), 16)  # 1024 → 16
        self.out = nn.Linear(16, 1)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: [bs, 64]
        bs, seq_len = tokens.shape
        assert seq_len == 64, f"Expected 64 squares, got {seq_len}"

        feats = self.embedding(tokens)  # [bs, 64, 24]
        q = feats[..., :4]       # [bs, 64, 4]
        k = feats[..., 4:8]      # [bs, 64, 4]
        v = feats[..., 8:16]     # [bs, 64, 8]
        e = feats[..., 16:]      # [bs, 64, 8] residual feats

        attn = torch.matmul(q, k.transpose(-2, -1)) / 2.0  # [bs, 64, 64]; scale √4=2
        attn = F.softmax(attn, dim=-1)

        z = torch.matmul(attn, v)  # [bs, 64, 8]

        combined = torch.cat([z, e], dim=-1)  # [bs, 64, 16]
        flat = combined.reshape(bs, -1)        # [bs, 1024]

        h = torch.relu(self.head(flat))        # [bs, 16]
        value = self.out(h)                    # [bs, 1]
        return value.squeeze(-1)               # [bs] logit
    
    def attn_compile(self) -> AttnModelCompiled:
        # Extract state_dict and convert to NumPy (CPU)
        state = self.state_dict()

        # Embedding: (960, 24) → direct
        embed_np = state['embedding.weight'].cpu().numpy()  # Shape: (960, 24)

        # Head Linear: in=1024, out=16 → weight (16, 1024) → transpose to (1024, 16)
        head_weight_np = state['head.weight'].cpu().numpy().T  # (1024, 16)
        head_bias_np = state['head.bias'].cpu().numpy()  # (16,)

        # Out Linear: in=16, out=1 → weight (1, 16) → transpose to (16, 1)
        out_weight_np = state['out.weight'].cpu().numpy().T  # (16, 1)
        out_bias_np = state['out.bias'].cpu().numpy()  # (1,)

        # Verify shapes (internal check; raises ValueError if mismatch)
        if embed_np.shape != (960, 24):
            raise ValueError(f"Embed shape mismatch: expected (960, 24), got {embed_np.shape}")
        if head_weight_np.shape != (1024, 16):
            raise ValueError(f"Head weight shape mismatch: expected (1024, 16), got {head_weight_np.shape}")
        if head_bias_np.shape != (16,):
            raise ValueError(f"Head bias shape mismatch: expected (16,), got {head_bias_np.shape}")
        if out_weight_np.shape != (16, 1):
            raise ValueError(f"Out weight shape mismatch: expected (16, 1), got {out_weight_np.shape}")
        if out_bias_np.shape != (1,):
            raise ValueError(f"Out bias shape mismatch: expected (1,), got {out_bias_np.shape}")

        # Create and return compiled model
        compiled = AttnModelCompiled(
            embed_np,          # (960, 24) float32
            head_weight_np,    # (1024, 16) float32
            head_bias_np,      # (16,) float32
            out_weight_np,     # (16, 1) float32
            out_bias_np        # (1,) float32
        )
        return compiled


if __name__ == "__main__":
    model = AttnModel()
    model.eval()  # Inference mode
    board = Board()
    board.set_start_position()
    
    tokens_torch = torch.Tensor(board.tokenize()).to(torch.int32).unsqueeze(0)

    summary(model, input_data=tokens_torch)

    scripted_model = torch.jit.script(model)

    # 3. FLOPs / Ops (Profiler: CPU time, calls, etc.)
    print("\nProfiler (FLOPs proxy via CPU ops/time):")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            for _ in range(100):  # Warmup/avg
                model(tokens_torch)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))

    # 4. Memory / Speed Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time
    start = time.time()
    for _ in range(10000):
        model(tokens_torch)
    elapsed = time.time() - start
    print(
        f"Torch Model: 10k forward calls: {elapsed:.3f}s total → {10000/elapsed:.0f} calls/sec"
    )
    print(f"10000 infs: {(time.time() - start)*1000:.0f} ms total → {1000/(time.time()-start):.0f} FPS")
    print(f"Peak mem: {torch.cuda.max_memory_allocated()/1e6:.1f} MB" if torch.cuda.is_available() else "CPU only")
    
    output = model(tokens_torch)
    print(f"Output: {output}")

    import time
    start = time.time()
    for _ in range(100000):
        tokens = board.tokenize()
    elapsed = time.time() - start
    print(
        f"100k board_to_tokens calls: {elapsed:.3f}s total → {100000/elapsed:.0f} calls/sec"
    )
    
    compiled_model = model.attn_compile()
    start = time.time()
    compiled_model.forward(tokens)
    for _ in range(10000):
        compiled_model.forward(tokens)
    elapsed = time.time() - start
    print(
        f"Cython Model: 10k forward calls: {elapsed:.3f}s total → {10000/elapsed:.0f} calls/sec"
    )
    print(output)
    
    
    
