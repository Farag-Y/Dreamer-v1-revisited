
from torch import nn, randn_like, tanh
from torch.nn import functional as F


class Actor(nn.Module):
  def __init__(self,state_size,hidden_size,action_size,non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.fc1 = nn.Linear(state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)

    self.mean_head = nn.Linear(hidden_size,action_size)
    self.std_head = nn.Linear(hidden_size,action_size)
  def forward(self, state):
      hidden = self.act_fn(self.fc1(state))
      hidden = self.act_fn(self.fc2(hidden))
      hidden = self.act_fn(self.fc3(hidden))
      mean = self.mean_head(hidden)
      mean = 5.0 * tanh(mean/5.0)
      std = F.softplus(self.std_head(hidden)) + 1e-4 
      return mean,std
  def sample(self,state):
     mean,std = self.forward(state)
     eps     = randn_like(mean)
     action  = tanh(mean + std * eps)
     return action               



class Critic(nn.Module):
  def __init__(self,state_size,hidden_size,non_linearity='relu'):
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.fc1 = nn.Linear(state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)
    self.fc4 = nn.linear(hidden_size,1)


  def forward(self, state):
    hidden = self.act_fn(self.fc1(state))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    value = self.act_fn(self.fc4(hidden))
    return value.squeeze(-1)