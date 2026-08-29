import os
from datetime import datetime

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from agent.dreamer import Dreamer
from checkpoint import load_checkpoint, save_checkpoint, save_experience_replay
from cloud_storage import upload_config
from env_wrapper import Env
from experience_replay import ExperienceReplay
from metrics import Metrics
from utils import collect_observations
from visualization import plot_metrics, write_video

'''
TODO:
1. [DONE] Remove the old CEM planner
2. Unify var nams between deterministic state of RSSM with 'belief'
3. Further clean ups
4. Add new losses to metrics
5. [DONE] update saving checkpoints and models
6. Improve complexity of v lambda computation.
7. Record episode reward into metrics.train_rewards during training
8. Move imagination horizon and gamma/lambda out of hardcoding and into cfg
9. Scope Dreamer.eval() in test() to match the original modules
10. Understand actor-then-critic vs critic-then-actor backward order (see docs/actor_critic_backward_fix.md)
'''

def _run_test_episode(cfg, env, dreamer):
    observation = env.reset()
    belief = torch.zeros(1, cfg.belief_size, device=dreamer.device)
    state  = torch.zeros(1, cfg.state_size,  device=dreamer.device)
    action = torch.zeros(1, env.action_size, device=dreamer.device)
    episode_reward, frames = 0.0, []
    for _ in range(cfg.max_episode_length // cfg.action_repeat):
        belief, state, action, observation, reward, done = dreamer.act(
            env, observation, belief, state, action, explore=False)
        episode_reward += reward
        frames.append(env.render_frame())
        if done:
            break
    return episode_reward, frames


def test(cfg: DictConfig, dreamer: Dreamer, env,
         metrics: Metrics, results_dir: str, episode: int = 0):
    dreamer.eval()
    episode_rewards, pad = [], len(str(cfg.test_episodes))
    with torch.no_grad():
        for ep_idx in tqdm(range(cfg.test_episodes), desc="Testing"):
            reward, frames = _run_test_episode(cfg, env, dreamer)
            episode_rewards.append(reward)
            ep_str = str(ep_idx).zfill(pad)
            write_video(frames, f'test_episode_{ep_str}', results_dir)
            cv2.imwrite(os.path.join(results_dir, f'test_episode_{ep_str}.png'),
                        frames[-1][:, :, ::-1])

    metrics.test_episodes.append(episode)
    metrics.test_rewards.append(episode_rewards)
    print(f"Average Test Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    plot_metrics(metrics, results_dir)
    metrics.save(os.path.join(results_dir, 'metrics.pt'))


def train(cfg: DictConfig, dreamer: Dreamer, experience_replay: ExperienceReplay,
          metrics: Metrics, env, results_dir: str, r2_prefix: str = ""):
    for episode in tqdm(range(metrics.last_episode + 1, cfg.episodes + 1), total=cfg.episodes, initial=metrics.last_episode):
        results = [dreamer.train_on_batch(experience_replay) for _ in tqdm(range(cfg.collect_interval))]
        metrics.record(results)
        dreamer.collect_episode(env, experience_replay, explore=True)
        plot_metrics(metrics, results_dir)
        if episode % cfg.checkpoint_interval == 0:
            save_checkpoint(cfg, episode, dreamer, metrics, results_dir, r2_prefix=r2_prefix)
        if cfg.experience_replay_interval and episode % cfg.experience_replay_interval == 0:
            save_experience_replay(cfg, episode, experience_replay, results_dir, r2_prefix=r2_prefix)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.test and not cfg.models:
        raise ValueError("Test mode requires a checkpoint: set cfg.models to a checkpoint path.")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if not cfg.disable_cuda and torch.cuda.is_available() else "cpu"
    env = Env(
        cfg.env,
        seed=cfg.seed,
        max_episode_length=cfg.max_episode_length,
        action_repeat=cfg.action_repeat,
        bit_depth=cfg.bit_depth,
    )

    dreamer = Dreamer.from_config(cfg, env, device)
    metrics = load_checkpoint(cfg, device, dreamer) if cfg.models else Metrics()

    if cfg.test:
        results_dir = os.path.join(hydra.utils.get_original_cwd(), 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(results_dir, exist_ok=True)
        test(cfg, dreamer, env, metrics, results_dir, episode=metrics.last_episode)
    else:
        experience_replay = (ExperienceReplay.load(cfg.experience_replay_path, device)
                             if cfg.experience_replay_path
                             else collect_observations(cfg, device, env, metrics))
        run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_dir = os.path.join(hydra.utils.get_original_cwd(), 'results', run_id)
        os.makedirs(results_dir, exist_ok=True)
        if getattr(cfg, 'r2_enabled', False):
            upload_config(cfg, run_id)
        train(cfg, dreamer, experience_replay, metrics, env, results_dir, r2_prefix=run_id)

    env.close()


if __name__ == "__main__":
    main()
