import os

import torch
from omegaconf import DictConfig

from agent.dreamer import Dreamer
from metrics import Metrics


def load_checkpoint(cfg: DictConfig, device: str, dreamer: Dreamer) -> Metrics:
    state = torch.load(cfg.models, map_location=device)
    dreamer.load_state_dict(state['model'])
    dreamer.world_model.optimizer.load_state_dict(state['world_model_optim'])
    dreamer.behavior.actor_optim.load_state_dict(state['actor_optim'])
    dreamer.behavior.critic_optim.load_state_dict(state['critic_optim'])
    return Metrics.load(os.path.join(os.path.dirname(cfg.models), 'metrics.pt'))


def save_checkpoint(cfg: DictConfig, episode: int, dreamer: Dreamer, metrics: Metrics,
                     results_dir: str, r2_prefix: str = "") -> None:
    checkpoint_dir = os.path.join(results_dir, f'checkpoint_{episode}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model':             dreamer.state_dict(),
        'world_model_optim': dreamer.world_model.optimizer.state_dict(),
        'actor_optim':       dreamer.behavior.actor_optim.state_dict(),
        'critic_optim':      dreamer.behavior.critic_optim.state_dict(),
    }, os.path.join(checkpoint_dir, 'models.pt'))
    metrics.save(os.path.join(checkpoint_dir, 'metrics.pt'))
    if getattr(cfg, 'r2_enabled', False):
        from cloud_storage import upload_checkpoint
        upload_checkpoint(cfg, checkpoint_dir, episode, r2_prefix)


def save_experience_replay(cfg: DictConfig, episode: int, experience_replay, results_dir: str, r2_prefix: str = "") -> None:
    replay_path = os.path.join(results_dir, f'experience_replay_{episode}.pt')
    experience_replay.save(replay_path)
    if getattr(cfg, 'r2_enabled', False):
        from cloud_storage import upload_experience_replay
        upload_experience_replay(cfg, replay_path, episode, r2_prefix)
