import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm


class FreezeParameters:
    '''
    Temporarily disables requires_grad on a module's parameters.

    Gradients can still flow *through* the module during backward (e.g. into
    an upstream actor), but the module's own weights won't accumulate .grad
    or get updated by its optimizer while frozen.
    '''

    def __init__(self, modules):
        self.modules = modules if isinstance(modules, (list, tuple)) else [modules]
        self.param_states = [p.requires_grad for m in self.modules for p in m.parameters()]

    def __enter__(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        i = 0
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(self.param_states[i])
                i += 1


def imagine_rollout(start_states, actor, rssm, H=15):
    pass


def compute_vlambda(states, rewards, critic, gamma=0.99, lam=0.95):
    pass


def critic_loss(states, v_lambda, critic):
    pass


def actor_loss(v_lambda):
    return -v_lambda.mean()


def execute_one_run_with_actor(cfg, device, env, rssm, encoder, actor, observation, belief, state, explore=True):
    pass


def train_actor_critic(cfg: DictConfig, device: str, env, actor, actor_optim, encoder, critic, critic_optim, rssm, experience_replay):
    '''
    1. Freeze RSSM.
    2. Imagination rollout.
    3. Compute V-lambda.
    4. Critic loss:
        A. Zero grad
        B. Loss
        C. Loss backward & step
    5. Actor loss:
        A. Zero grad
        B. Loss
        C. Loss backward & step
    6. Execute one run with actor.
    '''

    belief = torch.zeros(1, cfg.belief_size, device=device)  # TODO: needed?
    state = torch.zeros(1, cfg.state_size, device=device)
    observation = env.reset()
    episode_reward = 0.0

    with FreezeParameters(rssm):
        states, rewards = imagine_rollout(start_states=state, actor=actor, rssm=rssm)  # TODO: belief as input?
        v_lambda = compute_vlambda(states, rewards, critic)

    # Critic
    critic_optim.zero_grad()
    c_loss = critic_loss(states, v_lambda, critic)
    c_loss.backward()
    nn.utils.clip_grad_norm_(critic_optim.param_groups[0]['params'], cfg.grad_clip_norm)
    critic_optim.step()

    # Actor
    actor_optim.zero_grad()
    a_loss = actor_loss(v_lambda)
    a_loss.backward()
    nn.utils.clip_grad_norm_(actor_optim.param_groups[0]['params'], cfg.grad_clip_norm)
    actor_optim.step()

    # Update experience
    for _ in tqdm(range(cfg.max_episode_length // cfg.action_repeat)):
        belief, state, action, next_obs, reward, done = execute_one_run_with_actor(
            cfg, device, env, rssm, encoder, actor, observation, belief, state, explore=True)
        experience_replay.append(observation, reward, action.squeeze(0).cpu(), done)
        episode_reward += reward
        observation = next_obs
        if done:
            break
