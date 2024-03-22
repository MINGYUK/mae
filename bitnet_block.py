from timm.layers import Mlp
from timm.models.vision_transformer import Block
import torch.nn as nn
from torch.nn.modules import GELU, LayerNorm, Module


class bitBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0,
        attn_drop: float = 0,
        init_values: float | None = None,
        drop_path: float = 0,
        act_layer: Module = nn.GELU,
        norm_layer: Module = nn.LayerNorm,
        mlp_layer: Module = ...,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
