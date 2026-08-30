from types import TracebackType

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from env_wrapper import BaseEnv
from experience_replay import ExperienceReplay
from metrics import Metrics


def model_wrapper(model: torch.nn.Module, *inputs: torch.Tensor, trailing_dims: int = 3) -> torch.Tensor:
    leading = inputs[0].shape[:-trailing_dims]
    reshaped = [obs.reshape(-1, *obs.shape[-trailing_dims:]) for obs in inputs]
    out = model(*reshaped)
    return out.view(*leading, *out.shape[1:])


def preprocess_frame(frame: np.ndarray, size: int = 64) -> np.ndarray:
    frame = cv2.resize(frame, (size, size))
    frame = np.transpose(frame, (2, 0, 1))
    return frame.astype(np.float32) / 255.0


class FreezeParameters:
    '''
    Temporarily disables requires_grad on a module's parameters.

    Gradients can still flow *through* the module during backward (e.g. into
    an upstream actor), but the module's own weights won't accumulate .grad
    or get updated by its optimizer while frozen.
    '''

    def __init__(self, modules: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module, ...]) -> None:
        self.modules = modules if isinstance(modules, (list, tuple)) else [modules]
        self.param_states = [p.requires_grad for m in self.modules for p in m.parameters()]

    def __enter__(self) -> None:
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(False)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        i = 0
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(self.param_states[i])
                i += 1


def collect_observations(cfg: DictConfig, device: str, env: BaseEnv, metrics: Metrics) -> ExperienceReplay:
    experience_replay = ExperienceReplay(
        cfg.experience_size,
        observation_size=0,
        image_shape=list(env.observation_size),
        action_size=env.action_size,
        bit_depth=cfg.bit_depth,
        device=device,
    )
    for s in range(1, cfg.seed_episodes + 1):
        observation = env.reset()
        done = False
        while not done:
            action = env.sample_random_action()
            next_obs, reward, done = env.step(action)
            experience_replay.append(observation, reward, action, done)
            observation = next_obs
        metrics.steps.append(env.t + metrics.last_step)
        metrics.episodes.append(metrics.last_episode + 1)
    return experience_replay
