from typing import Callable, Optional

import torch.nn.functional as F
import torch
from torch import nn, Tensor
from bitnet.nn.bitlinear import BitLinear


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps: float = 1e-5
        self.quantization_range: int = 2 ** (num_bits - 1)  # Q_b in the paper

    def ste_weights(self, weights_gamma: float) -> Tensor:
        eps: float = 1e-7
        scaled_weights: Tensor = self.weight / (weights_gamma + eps)
        bin_weights_no_grad: Tensor = torch.clamp(
            torch.round(scaled_weights), min=-1, max=1
        )
        bin_weights_with_grad: Tensor = (
            bin_weights_no_grad - self.weight
        ).detach() + self.weight
        return bin_weights_with_grad

    def binarize_weights(self, weights_gamma: float) -> Tensor:
        binarized_weights = self.ste_weights(weights_gamma)
        return binarized_weights

    def quantize_activations(self, _input: Tensor, input_gamma: float) -> Tensor:
        # Equation 4 BitNet paper
        quantized_input = torch.clamp(
            _input * self.quantization_range / input_gamma,
            -self.quantization_range + self.eps,
            self.quantization_range - self.eps,
        )
        return quantized_input

    def dequantize_activations(
        self, _input: Tensor, input_gamma: float, beta: float
    ) -> Tensor:
        return _input * input_gamma * beta / self.quantization_range

    def forward(self, _input: Tensor) -> Tensor:
        normalized_input: Tensor = nn.functional.layer_norm(_input, (_input.shape[1:]))
        input_gamma: float = normalized_input.abs().max().item()
        weight_abs_mean: float = self.weight.abs().mean().item()

        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = torch.nn.functional.linear(input_quant, binarized_weights, self.bias)
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)

        return output


def default(val, d):
    return val if val is not None else d


def init_zero_(tensor):
    nn.init.constant_(tensor, 0.0)


# [GLU]
class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
        linear (Callable, optional): Linear function to be used for projection. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable,
        mult_bias: bool = False,
        linear: Callable = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        self.mult_bias = mult_bias

        if linear:
            self.proj = linear(dim_in, dim_out * 2)
        else:
            self.proj = BitLinear(dim_in, dim_out, bias=self.mult_bias)

        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate) * self.mult_bias


# [FEATURE] Add type hints to the forward method
class BitFeedForward(nn.Module):
    """
    BitFeedForward module performs feed-forward operations on the input tensor.

    Args:
        dim (int): The input dimension.
        dim_out (int, optional): The output dimension. If not provided, it is set to the input dimension.
        mult (int, optional): The multiplier for the inner dimension. Default is 4.
        glu (bool, optional): Whether to use Gated Linear Unit (GLU) activation. Default is False.
        glu_mult_bias (bool, optional): Whether to apply bias to the GLU activation. Default is False.
        swish (bool, optional): Whether to use Swish activation. Default is False.
        relu_squared (bool, optional): Whether to use squared ReLU activation. Default is False.
        post_act_ln (bool, optional): Whether to apply Layer Normalization after activation. Default is False.
        dropout (float, optional): The dropout probability. Default is 0.0.
        no_bias (bool, optional): Whether to exclude bias in linear layers. Default is False.
        zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Default is False.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        swish: bool = False,
        # post_act_ln: bool = False,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        self.ff = nn.Sequential(
            BitLinear(dim, inner_dim, bias=not no_bias),
            activation,
            # nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            BitLinear(inner_dim, dim_out, bias=not no_bias),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the BitFeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.ff(x)
