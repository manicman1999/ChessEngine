import chess
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_PIECES = 6
GROUP_SIZE = NUM_PIECES * 2 + 1  # 13: 12 pieces + empty
VOCAB_SIZE = GROUP_SIZE * 64  # 832
EMBED_DIM = 32

class AttnModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=4, ff_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # No padding_idx
        
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, ff_dim, batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)
        )
    
    def forward(self, tokens):  # [B, 64] token ids
        x = self.embed(tokens)
        x = self.transformer(x)
        x = self.pool(x.transpose(1,2)).squeeze(-1)
        return self.fc(x).squeeze(-1)

def board_to_tokens(board: chess.Board) -> torch.Tensor:
    tokens = torch.zeros(64, dtype=torch.long)
    for sq in range(64):
        piece = board.piece_at(sq)
        offset = sq * GROUP_SIZE
        if piece:
            pt_idx = PIECE_TYPES.index(piece.piece_type)
            col_off = NUM_PIECES if piece.color == chess.BLACK else 0
            tokens[sq] = offset + pt_idx + col_off
        else:
            tokens[sq] = offset + 12  # Empty per sq
    return tokens.unsqueeze(0)


if __name__ == "__main__":
    model = AttnModel()
    model.eval()  # Inference mode
    tokens = board_to_tokens(chess.Board())  # Dummy input [1,64]

    summary(model, input_data=tokens, verbose=2, col_names=["input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])

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
                model(tokens)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))

    # 4. Memory / Speed Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time
    start = time.time()
    for _ in range(10000):
        model(tokens)
    print(f"10000 infs: {(time.time() - start)*1000:.0f} ms total → {1000/(time.time()-start):.0f} FPS")
    print(f"Peak mem: {torch.cuda.max_memory_allocated()/1e6:.1f} MB" if torch.cuda.is_available() else "CPU only")