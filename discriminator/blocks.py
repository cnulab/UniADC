import torch
import torch.nn as nn
import math

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, channel, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            channel,
            channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(channel)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        channel,
        activation,
        bn=False,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.groups = 1
        self.out_conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )
        self.resConfUnit1 = ResidualConvUnit_custom(channel, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(channel, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = self.out_conv(output)
        return output