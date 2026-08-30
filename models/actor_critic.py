
import torch
from torch import nn, randn_like, tanh
from torch.nn import functional as F


class Actor(nn.Module):
  def __init__(
      self,
      belief_size: int,
      state_size: int,
      hidden_size: int,
      action_size: int,
      non_linearity: str = 'relu',
  ) -> None:
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.fc1 = nn.Linear(belief_size+state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)

    self.mean_head = nn.Linear(hidden_size,action_size)
    self.std_head = nn.Linear(hidden_size,action_size)
  def forward(self, belief: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      hidden = self.act_fn(self.fc1(torch.concat((belief,state),dim=1)))
      hidden = self.act_fn(self.fc2(hidden))
      hidden = self.act_fn(self.fc3(hidden))
      mean = self.mean_head(hidden)
      mean = 5.0 * tanh(mean/5.0)
      std = F.softplus(self.std_head(hidden)) + 1e-4
      return mean,std
  def sample(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
     mean,std = self.forward(belief,state)
     eps     = randn_like(mean)
     action  = tanh(mean + std * eps)
     return action
  def mode(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    mean,_ = self.forward(belief,state)
    return tanh(mean)



class Critic(nn.Module):
  def __init__(
      self,
      belief_size: int,
      state_size: int,
      hidden_size: int,
      non_linearity: str = 'relu',
  ) -> None:
    super().__init__()
    self.act_fn = getattr(F, non_linearity)
    self.fc1 = nn.Linear(belief_size+state_size,hidden_size)
    self.fc2 = nn.Linear(hidden_size,hidden_size)
    self.fc3 = nn.Linear(hidden_size,hidden_size)
    self.fc4 = nn.Linear(hidden_size,1)


  def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    hidden = self.act_fn(self.fc1(torch.concat((belief,state),dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    value = self.fc4(hidden)
    return value.squeeze(-1)

