import os

import matplotlib.pyplot as plt

from metrics import Metrics


def plot_metrics(metrics: Metrics, results_dir: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Training Metrics — Episode {metrics.last_episode}')

    if metrics.kl_loss:
        axes[0, 0].plot(metrics.kl_loss)
        axes[0, 0].set_title('KL Loss')
        axes[0, 0].set_xlabel('Episode')

    if metrics.observation_loss:
        axes[0, 1].plot(metrics.observation_loss)
        axes[0, 1].set_title('Observation Loss')
        axes[0, 1].set_xlabel('Episode')

    if metrics.reward_loss:
        axes[0, 2].plot(metrics.reward_loss)
        axes[0, 2].set_title('Reward Loss')
        axes[0, 2].set_xlabel('Episode')

    if metrics.actor_loss:
        axes[1, 0].plot(metrics.actor_loss)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Episode')

    if metrics.critic_loss:
        axes[1, 1].plot(metrics.critic_loss)
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Episode')

    if metrics.discount_loss:
        axes[1, 2].plot(metrics.discount_loss)
        axes[1, 2].set_title('Discount Loss')
        axes[1, 2].set_xlabel('Episode')

    if metrics.train_rewards:
        axes[2, 0].plot(metrics.train_rewards)
        axes[2, 0].set_title('Episode Reward')
        axes[2, 0].set_xlabel('Episode')

    if metrics.test_rewards:
        avg_test = [sum(ep) / len(ep) for ep in metrics.test_rewards]
        axes[2, 1].plot(metrics.test_episodes, avg_test)
        axes[2, 1].set_title('Avg Test Reward')
        axes[2, 1].set_xlabel('Episode')
    else:
        axes[2, 1].axis('off')

    if metrics.train_env_steps:
        axes[2, 2].plot(metrics.train_env_steps, metrics.train_rewards)
        axes[2, 2].set_title('Episode Reward')
        axes[2, 2].set_xlabel('Environment Steps')
    else:
        axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics.png'))
    plt.close(fig)


def write_video(frames: list, title: str, path: str, fps: int = 30) -> None:
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(path, f'{title}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h),
    )
    for frame in frames:
        writer.write(frame[:, :, ::-1])
    writer.release()
