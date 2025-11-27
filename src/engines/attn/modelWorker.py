import time
import torch
import asyncio
from math import inf


class ModelWorker:

    def __init__(self, model: torch.nn.Module, batch_size: int = 32):
        self.model = model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.eval_queue = asyncio.Queue(maxsize=1024)
        self.last_flush_time = time.time()
        self.worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        batch = []
        while True:
            try:
                item = await asyncio.wait_for(self.eval_queue.get(), timeout=0.001)
                future, input_tensor = item
                batch.append((future, input_tensor))
                now = time.time()
                if len(batch) >= self.batch_size or (
                    len(batch) > 0 and (now - self.last_flush_time > 0.5)
                ):
                    await self._process_batch(batch)
                    batch = []
                    self.last_flush_time = now
                self.eval_queue.task_done()
            except asyncio.TimeoutError:
                if batch:
                    await self._process_batch(batch)
                    batch = []
                continue

    async def _process_batch(self, batch):
        if not batch:
            return
        valid_batch = [(future, inp) for future, inp in batch if not future.cancelled()]
        if not valid_batch:
            return
        futures = [future for future, _ in valid_batch]
        batch_inputs = torch.cat([inp for _, inp in valid_batch]).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_inputs)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1).cpu().tolist()
            else:
                outputs = outputs.cpu().tolist()
        for i, future in enumerate(futures):
            future.set_result(outputs[i])

    async def eval(self, input_tensor: torch.Tensor) -> float:
        if input_tensor.shape[0] != 1:
            raise ValueError("Input must be single sample, e.g., [1, ...]")
        input_tensor = torch.clone(input_tensor.to("cpu"))
        future = asyncio.Future()
        await self.eval_queue.put((future, input_tensor))
        return await future

    def __del__(self):
        self.worker_task.cancel()
