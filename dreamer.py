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
        #TODO: Re-align on rewards, actions, and state allignment
        rewards.append(reward_model(belief, state))  # r_tau ~ q(r_tau | s_tau), decoded before the step

        action = actor.sample(belief, state).unsqueeze(0)
        rssm_out = rssm(state, action, belief)
        belief = rssm_out.det_hidden_states[-1]
        state = rssm_out.prior_states[-1]

        states.append(state)
        beliefs.append(belief)

    states = torch.stack(states, dim=1)    # [batch, H+1, latent_dim]
    beliefs = torch.stack(beliefs, dim=1)  # [batch, H+1, belief_dim]
    rewards = torch.stack(rewards, dim=1)  # [batch, H]
    return states, beliefs, rewards

def compute_vlambda(states, beliefs, rewards, critic, gamma=0.99, lam=0.95):
    '''
    TD(lambda) targets for an imagined rollout (Dreamer v1, eqs. 6-7).

    states, beliefs: [batch, H+1, dim], indexed tau = 0..H.
    rewards:         [batch, H],        indexed tau = 0..H-1, rewards[:, tau] = r_tau ~ q(r_tau|s_tau).

    We flatten states/beliefs to [batch*(H+1), dim] since the critic (like the actor) expects [batch, dim].
    '''
    batch, H = rewards.shape
    device, dtype = states.device, states.dtype

    with FreezeParameters(critic):
        values = critic(
            beliefs.reshape(-1, beliefs.shape[-1]),
            states.reshape(-1, states.shape[-1]),
        ).reshape(batch, H + 1)

    gamma_pows = gamma ** torch.arange(H + 1, device=device, dtype=dtype)

    V_lambda = torch.zeros(batch, H + 1, device=device, dtype=dtype)
    V_lambda[:, H] = values[:, H]

    for tau in range(H):
        max_k = H - tau 
        vl = torch.zeros(batch, device=device, dtype=dtype)
        r_sum = torch.zeros(batch, device=device, dtype=dtype)

        for k in range(1, max_k + 1):
            r_sum = r_sum + gamma_pows[k - 1] * rewards[:, tau + k - 1]
            V_kN = r_sum + gamma_pows[k] * values[:, tau + k]

            is_final = k == max_k
            weight = lam ** (k - 1) if is_final else (1 - lam) * lam ** (k - 1)
            vl = vl + weight * V_kN

        V_lambda[:, tau] = vl
    return V_lambda

def critic_loss(states, beliefs, v_lambda, critic):
    batch, T = states.shape[:2]  # T = H + 1 (tau = 0..H, matches v_lambda)
    v_pred = critic(
        beliefs.reshape(-1, beliefs.shape[-1]),
        states.reshape(-1, states.shape[-1]),
    ).reshape(batch, T)
    target = v_lambda.detach()
    return 0.5 * (v_pred - target).pow(2).mean()


def actor_loss(v_lambda):
    return -v_lambda.mean()


def execute_one_run_with_actor(cfg, device, env, rssm, encoder, actor, observation, belief, state,action,explore=True):

    with torch.no_grad():
        encoded = encoder(observation.to(device))
        rssm_out = rssm(state, action.unsqueeze(0), belief, encoded.unsqueeze(0)) ## action has to go through RSSM first ? 
        belief = rssm_out.det_hidden_states[-1]
        state  = rssm_out.posterior_states[-1]
        action = actor.mode(belief,state) 

        action = action + torch.randn_like(action)*0.3 if explore else action
        action=action.clamp(cfg.min_action,cfg.max_action)
        next_obs, reward, done = env.step(action[0].cpu())
    return belief, state, action, next_obs, reward, done

def update_experience(cfg,env,rssm,encoder,actor,experience_replay,device):
    belief = torch.zeros(1, cfg.belief_size, device=device)
    state  = torch.zeros(1, cfg.state_size,  device=device)
    action = torch.zeros(1, env.action_size, device=device)
    observation = env.reset()
    episode_reward = 0.0
    for _ in tqdm(range(cfg.max_episode_length // cfg.action_repeat)):
        belief, state, action, next_obs, reward, done = execute_one_run_with_actor(
            cfg, device, env, rssm, encoder, actor, observation, belief, state,action, explore=True)
        experience_replay.append(observation, reward, action.squeeze(0).cpu(), done)
        episode_reward += reward
        observation = next_obs
        if done:
            break
def train_actor_critic(cfg: DictConfig, device: str,state,belief, env, actor, actor_optim, encoder, critic, critic_optim, rssm, reward_model, experience_replay):


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

