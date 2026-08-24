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


def imagine_rollout(start_state, start_belief, actor, reward_model, rssm, H=15):
    states = [start_state]
    beliefs = [start_belief]
    rewards = []

    state = start_state
    belief = start_belief
    for _ in range(H):
        action = actor.sample(belief, state).unsqueeze(0)
        rssm_out = rssm(state, action, belief)
        belief = rssm_out.det_hidden_states[-1]
        state = rssm_out.prior_states[-1]
        reward = reward_model(belief, state)

        states.append(state)
        beliefs.append(belief)
        rewards.append(reward)

    states = torch.stack(states, dim=1)    # [batch, H+1, latent_dim]
    beliefs = torch.stack(beliefs, dim=1)  # [batch, H+1, belief_dim]
    rewards = torch.stack(rewards, dim=1)  # [batch, H]
    return states, beliefs, rewards


def compute_vlambda(states, beliefs, rewards, critic, gamma=0.99, lam=0.95):
    #States and beliefs are [batch, H+1, dim], we will need to squeeze it into shape [B*T,dim] as the critic ( and also the actor) expects the shapes as [batch,dim]
    H = rewards.shape[1]
    batch = rewards.shape[0]
    values = critic (beliefs.reshape(-1,beliefs.shape[-1]),states.reshape(-1,states.shape[-1]),)

def critic_loss(states, beliefs, v_lambda, critic):
    pass


def actor_loss(v_lambda):
    return -v_lambda.mean()


def execute_one_run_with_actor(cfg, device, env, rssm, encoder, actor, observation, belief, state, explore=True):
    pass


def train_actor_critic(cfg: DictConfig, device: str, env, actor, actor_optim, encoder, critic, critic_optim, rssm, reward_model, experience_replay):
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
        states, beliefs, rewards = imagine_rollout(
            start_state=state, start_belief=belief, actor=actor, reward_model=reward_model, rssm=rssm)
        v_lambda = compute_vlambda(states, beliefs, rewards, critic)

    # Critic
    critic_optim.zero_grad()
    c_loss = critic_loss(states, beliefs, v_lambda, critic)
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
