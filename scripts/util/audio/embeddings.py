import torch.nn as nn
from einops import rearrange


class Embeddings(nn.Module):
    def __init__(self, code_size=1280, merge_type="cat") -> None:
        super().__init__()
        self.code_size = code_size * 2 if merge_type == "cat" else code_size
        self.merge_type = merge_type

    def forward(self, x):
        if self.merge_type == "cat":
            if x.dim() == 3:
                return rearrange(x, "b d c -> b (d c)")
            return rearrange(x, "b f d c -> b f (d c)")
        elif self.merge_type == "sum":
            return x.sum(dim=-2)
        elif self.merge_type == "mean":
            return x.mean(dim=-2)
        elif self.merge_type == "None":
            return x
        else:
            raise NotImplementedError
