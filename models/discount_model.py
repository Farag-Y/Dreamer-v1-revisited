

import torch
from torch import nn
from torch.nn import functional as F


class DiscountModel(nn.Module):
  def __init__(
      self,
      belief_size: int,
      state_size: int,
      hidden_size:int,
      non_linearity: str = 'relu',#TODO: ELU instead of relu for dreamer ?
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
  
