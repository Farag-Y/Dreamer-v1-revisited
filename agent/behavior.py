from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch import nn, optim

from env_wrapper import TERMINATING_ENVS
from models.actor_critic import Actor, Critic
from models.discount_model import DiscountModel
from models.reward_model import RewardModel
from models.rssm import RSSM
from utils import FreezeParameters

if TYPE_CHECKING:
    from agent.world_model import WorldModel


def _imagine_rollout(
    start_state: torch.Tensor,
    start_belief: torch.Tensor,
    actor: Actor,
    reward_model: RewardModel,
    rssm: RSSM,
    discount_model:DiscountModel,
    horizon: int = 15,
    discount_model_gamma:float=0.99,
    discount_enabled:bool=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [start_state]
    beliefs = [start_belief]
    rewards = []
    discounts=[]

    state = start_state
    belief = start_belief
    for _ in range(horizon):
        rewards.append(reward_model(belief, state))

        action = actor.sample(belief, state).unsqueeze(0)
        rssm_out = rssm(state, action, belief)
        gamma_hat = (
            torch.sigmoid(discount_model(belief, state)) * discount_model_gamma
            if discount_enabled
            else torch.full_like(rewards[-1], discount_model_gamma)
        )

        belief = rssm_out.det_hidden_states[-1]
        state = rssm_out.prior_states[-1]

        states.append(state)
        beliefs.append(belief)
        discounts.append(gamma_hat)

    states = torch.stack(states, dim=1)
    beliefs = torch.stack(beliefs, dim=1)
    rewards = torch.stack(rewards, dim=1)
    discounts = torch.stack(discounts,dim=1)
    return states, beliefs, rewards,discounts
#TODO: Revise again calculation of vlambda !
def _compute_vlambda(
    states: torch.Tensor,
    beliefs: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    critic: Critic,
    lam: float = 0.95,
) -> torch.Tensor:
    batch, H = rewards.shape
    device, dtype = states.device, states.dtype

    with FreezeParameters(critic):
        values = critic(
            beliefs.reshape(-1, beliefs.shape[-1]),
            states.reshape(-1, states.shape[-1]),
        ).reshape(batch, H + 1)

    V_lambda = torch.zeros(batch, H + 1, device=device, dtype=dtype)
    V_lambda[:, H] = values[:, H]

    for tau in range(H):
        max_k = H - tau
        vl = torch.zeros(batch, device=device, dtype=dtype)
        r_sum = torch.zeros(batch, device=device, dtype=dtype)
        discount_prod = torch.ones(batch, device=device, dtype=dtype)  # prod_{i=tau}^{tau+k-2} discounts[:,i]

        for k in range(1, max_k + 1):
            r_sum = r_sum + discount_prod * rewards[:, tau + k - 1]
            discount_prod = discount_prod * discounts[:, tau + k - 1]  # now prod over k terms
            V_kN = r_sum + discount_prod * values[:, tau + k]

            is_final = k == max_k
            weight = lam ** (k - 1) if is_final else (1 - lam) * lam ** (k - 1)
            vl = vl + weight * V_kN

        V_lambda[:, tau] = vl
    return V_lambda


def _critic_loss(
    states: torch.Tensor, beliefs: torch.Tensor, v_lambda: torch.Tensor, critic: Critic,
) -> torch.Tensor:
    batch, T = states.shape[:2]
    v_pred = critic(
        beliefs.detach().reshape(-1, beliefs.shape[-1]),
        states.detach().reshape(-1, states.shape[-1]),
    ).reshape(batch, T)
    target = v_lambda.detach()
    return 0.5 * (v_pred - target).pow(2).mean()


def _actor_loss(v_lambda: torch.Tensor) -> torch.Tensor:
    return -v_lambda.mean()


class ActorCritic(nn.Module):
    def __init__(self, cfg: DictConfig, action_size: int, device: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor = Actor(
            belief_size=cfg.belief_size,
            state_size=cfg.state_size,
            hidden_size=cfg.hidden_size,
            action_size=action_size,
            non_linearity=cfg.activation_function,
        ).to(device=device)
        self.critic = Critic(
            belief_size=cfg.belief_size,
            state_size=cfg.state_size,
            hidden_size=cfg.hidden_size,
            non_linearity=cfg.activation_function,
        ).to(device=device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate, eps=cfg.adam_epsilon)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_learning_rate, eps=cfg.adam_epsilon)
        self.discount_enabled = cfg.env in TERMINATING_ENVS


    def act(self, belief: torch.Tensor, state: torch.Tensor, explore: bool) -> torch.Tensor:
        action = self.actor.mode(belief, state)
        if explore:
            action = action + self.cfg.action_noise * torch.randn_like(action)
        return action

    def train_step(self, state: torch.Tensor, belief: torch.Tensor, world_model: "WorldModel") -> dict[str, float]:
        cfg = self.cfg
        state = state.reshape(-1, state.shape[-1]).detach()
        belief = belief.reshape(-1, belief.shape[-1]).detach()

        with FreezeParameters(world_model.rssm):
            states, beliefs, rewards,discounts = _imagine_rollout(
                start_state=state, start_belief=belief,
                actor=self.actor, reward_model=world_model.reward_model, rssm=world_model.rssm,discount_model=world_model.discount_model,
                horizon=cfg.imagination_horizon,discount_model_gamma=cfg.discount_model_gamma,discount_enabled=self.discount_enabled
            )
            v_lambda = _compute_vlambda(states, beliefs, rewards, discounts, self.critic, cfg.lam)

        self.actor_optim.zero_grad()
        a_loss = _actor_loss(v_lambda)
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_optim.param_groups[0]['params'], cfg.grad_clip_norm)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        c_loss = _critic_loss(states, beliefs, v_lambda, self.critic)
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_optim.param_groups[0]['params'], cfg.grad_clip_norm)
        self.critic_optim.step()

        return {'actor_loss': a_loss.item(), 'critic_loss': c_loss.item()}
