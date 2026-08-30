import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

from models.encoder import Encoder
from models.observation_model import ObservationModel
from models.reward_model import RewardModel
from models.rssm import RSSM, RSSMOutput
from utils import model_wrapper


def _latent_overshooting(
    cfg: DictConfig,
    rssm: RSSM,
    reward_model: RewardModel,
    actions: torch.Tensor,
    nonterminals: torch.Tensor,
    posterior_states: torch.Tensor,
    posterior_means: torch.Tensor,
    posterior_std_devs: torch.Tensor,
    rssm_beliefs: torch.Tensor,
    rewards: torch.Tensor,
    free_nats: torch.Tensor,
    device: str,
) -> torch.Tensor:
    if cfg.overshooting_kl_beta == 0:
        return torch.tensor(0.0, device=device)

    chunk_size = actions.shape[0] + 1
    B, state_size = posterior_states.shape[1], posterior_states.shape[2]

    overshooting_vars = []
    for t in range(1, chunk_size - 1):
        d = min(t + cfg.overshooting_distance, chunk_size - 1)
        t_ = t - 1
        pad_len = cfg.overshooting_distance - (d - t)
        seq_pad = (0, 0, 0, 0, 0, pad_len)
        overshooting_vars.append((
            F.pad(actions[t:d], seq_pad),
            F.pad(nonterminals[t:d], seq_pad),
            F.pad(rewards[t:d], seq_pad[2:]),
            rssm_beliefs[t_],
            posterior_states[t_].detach(),
            F.pad(posterior_means[t:d].detach(), seq_pad),
            F.pad(posterior_std_devs[t:d].detach(), seq_pad, value=1),
            F.pad(torch.ones(d - t, B, state_size, device=device), seq_pad),
        ))

    overshooting_vars = tuple(zip(*overshooting_vars))

    prior_out = rssm(
        torch.cat(overshooting_vars[4], dim=0),
        torch.cat(overshooting_vars[0], dim=1),
        torch.cat(overshooting_vars[3], dim=0),
        None,
        torch.cat(overshooting_vars[1], dim=1),
    )

    seq_mask     = torch.cat(overshooting_vars[7], dim=1)
    target_means = torch.cat(overshooting_vars[5], dim=1)
    target_stds  = torch.cat(overshooting_vars[6], dim=1)

    kl = (kl_divergence(
        Normal(target_means, target_stds),
        Normal(prior_out.prior_means, prior_out.prior_std_devs),
    ) * seq_mask).sum(dim=2)
    total = (1 / cfg.overshooting_distance) * cfg.overshooting_kl_beta * \
        torch.max(kl, free_nats).mean(dim=(0, 1)) * (chunk_size - 1)

    if cfg.overshooting_reward_scale != 0:
        target_rewards = torch.cat(overshooting_vars[2], dim=1)
        pred_rewards = model_wrapper(reward_model, prior_out.det_hidden_states, prior_out.prior_states, trailing_dims=1)
        reward_mask = seq_mask[:, :, 0]
        total = total + (1 / cfg.overshooting_distance) * cfg.overshooting_reward_scale * \
            F.mse_loss(pred_rewards * reward_mask, target_rewards, reduction='none').mean(dim=(0, 1)) * (chunk_size - 1)

    return total


class WorldModel(nn.Module):
    def __init__(self, cfg: DictConfig, action_size: int, device: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.rssm = RSSM(
            state_size=cfg.state_size,
            hidden_size=cfg.hidden_size,
            belief_size=cfg.belief_size,
            action_size=action_size,
            obs_size=cfg.embedding_size,
            non_linearity=cfg.activation_function,
        ).to(device=device)
        self.decoder = ObservationModel(
            belief_size=cfg.belief_size, state_size=cfg.state_size, embedding_size=cfg.embedding_size,
        ).to(device=device)
        self.reward_model = RewardModel(
            belief_size=cfg.belief_size, state_size=cfg.state_size, hidden_size=cfg.hidden_size,
        ).to(device=device)
        self.encoder = Encoder(embedding_size=cfg.embedding_size).to(device=device)
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    def observe(
        self,
        actions: torch.Tensor,
        encoded_obs: torch.Tensor,
        init_belief: torch.Tensor,
        init_state: torch.Tensor,
        nonterminals: torch.Tensor | None = None,
    ) -> RSSMOutput:
        return self.rssm(init_state, actions, init_belief, encoded_obs, nonterminals)

    def compute_loss(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, nonterminals: torch.Tensor,
    ) -> tuple[torch.Tensor, RSSMOutput, dict[str, float]]:
        cfg, device = self.cfg, self.device
        init_belief = torch.zeros(cfg.batch_size, cfg.belief_size, device=device)
        init_state  = torch.zeros(cfg.batch_size, cfg.state_size,  device=device)
        free_nats = torch.full((1,), cfg.free_nats, dtype=torch.float32, device=device)

        encoded_obs = model_wrapper(self.encoder, obs[1:])
        rssm_output: RSSMOutput = self.observe(actions[:-1], encoded_obs, init_belief, init_state, nonterminals[:-1])
        predicted_reward = model_wrapper(self.reward_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)

        kl_div = kl_divergence(
            Normal(rssm_output.posterior_means, rssm_output.posterior_std_devs),
            Normal(rssm_output.prior_means,     rssm_output.prior_std_devs),
        ).sum(dim=-1)
        kl_loss = torch.max(kl_div, free_nats).mean()

        decoded_obs = model_wrapper(self.decoder, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)
        obs_loss    = F.mse_loss(decoded_obs, obs[1:], reduction='none').sum((2, 3, 4)).mean()
        reward_loss = F.mse_loss(predicted_reward, rewards[:-1], reduction='none').mean()

        overshooting_loss = _latent_overshooting(
            cfg, self.rssm, self.reward_model,
            actions[:-1], nonterminals[:-1],
            rssm_output.posterior_states,
            rssm_output.posterior_means,
            rssm_output.posterior_std_devs,
            rssm_output.det_hidden_states,
            rewards,
            free_nats,
            device,
        )

        total_loss = kl_loss + obs_loss + reward_loss + overshooting_loss
        loss_components = {
            'kl_loss': kl_loss.item(),
            'obs_loss': obs_loss.item(),
            'reward_loss': reward_loss.item(),
            'overshooting_loss': overshooting_loss.item(),
        }
        return total_loss, rssm_output, loss_components

    def train_step(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, nonterminals: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        self.optimizer.zero_grad()
        total_loss, rssm_output, loss_components = self.compute_loss(obs, actions, rewards, nonterminals)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.cfg.grad_clip_norm)
        self.optimizer.step()
        return {
            **loss_components,
            'belief': rssm_output.det_hidden_states[-1],
            'state': rssm_output.posterior_states[-1],
        }
