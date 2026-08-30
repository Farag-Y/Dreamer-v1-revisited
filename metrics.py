from dataclasses import dataclass, field

import torch


@dataclass
class Metrics:
    steps:            list[int]         = field(default_factory=list)
    episodes:         list[int]         = field(default_factory=list)
    train_rewards:    list[float]       = field(default_factory=list)
    test_episodes:    list[int]         = field(default_factory=list)
    test_rewards:     list[list[float]] = field(default_factory=list)
    observation_loss: list[float]       = field(default_factory=list)
    reward_loss:      list[float]       = field(default_factory=list)
    kl_loss:              list[float]       = field(default_factory=list)
    overshooting_loss:    list[float]       = field(default_factory=list)
    discount_loss:        list[float]       = field(default_factory=list)

    @property
    def last_episode(self) -> int:
        return self.episodes[-1] if self.episodes else 0

    @property
    def last_step(self) -> int:
        return self.steps[-1] if self.steps else 0

    def record(self, results: list[dict]) -> None:
        n = len(results)
        self.kl_loss.append(sum(r['kl_loss'] for r in results) / n)
        self.observation_loss.append(sum(r['obs_loss'] for r in results) / n)
        self.reward_loss.append(sum(r['reward_loss'] for r in results) / n)
        # self.overshooting_loss.append(sum(r['overshooting_loss'] for r in results) / n)
        self.discount_loss.append(sum(r['discount_loss'] for r in results) / n)

    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "Metrics":
        import numpy as np
        import numpy.core.multiarray
        import numpy.dtypes
        safe = [cls, numpy.core.multiarray._reconstruct, numpy.core.multiarray.scalar,
                np.ndarray, np.dtype, numpy.dtypes.Float32DType, numpy.dtypes.Float64DType]
        with torch.serialization.safe_globals(safe):
            return torch.load(path, weights_only=True)
