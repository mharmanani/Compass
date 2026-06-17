import numpy as np
import torch


from dataclasses import dataclass


@dataclass
class SelectFrameSubsequence:
    mode: str = "uniform"
    n_frames: int = 16
    dim: int = 1
    jitter_strength: float = 0 

    def __call__(self, sweep, *synchronized_sequences):
        assert isinstance(sweep, torch.Tensor)
        sample_positions = self.get_indices(sweep)

        sweep = sweep.index_select(
            self.dim, torch.tensor(sample_positions).to(sweep.device)
        )
        outputs = [sweep]
        for s in synchronized_sequences: 
            outputs.append(s.index_select(self.dim, torch.tensor(sample_positions)))

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def get_indices(self, sweep): 
        
        N = sweep.shape[self.dim]
        
        if self.mode == "uniform":
            uniform_sample_positions = np.linspace(0, N - 1, self.n_frames).astype(int)
            if self.jitter_strength > 0:
                jitter = np.random.uniform(
                    -self.jitter_strength, self.jitter_strength, size=uniform_sample_positions.shape
                )
                jitter = (jitter * (N / self.n_frames)).astype(int)
                uniform_sample_positions = np.clip(
                    uniform_sample_positions + jitter, 0, N - 1
                ).astype(int)
            return uniform_sample_positions
        else:
            raise ValueError(f"Unknown subsequence selection mode: {self.mode}")