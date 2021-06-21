import torch.nn as nn
import torch

from src.backbones.convlstm import ConvLSTM


class FPNConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        inconv=[32, 64],
        n_levels=5,
        n_channels=64,
        hidden_size=88,
        input_shape=(128, 128),
        mid_conv=True,
        pad_value=0,
    ):
        """
        Feature Pyramid Network with ConvLSTM baseline.
        Args:
            input_dim (int): Number of channels in the input images.
            num_classes (int): Number of classes.
            inconv (List[int]): Widths of the input convolutional layers.
            n_levels (int): Number of different levels in the feature pyramid.
            n_channels (int): Number of channels for each channel of the pyramid.
            hidden_size (int): Hidden size of the ConvLSTM.
            input_shape (int,int): Shape (H,W) of the input images.
            mid_conv (bool): If True, the feature pyramid is fed to a convolutional layer
            to reduce dimensionality before being given to the ConvLSTM.
            pad_value (float): Padding value (temporal) used by the dataloader.
        """
        super(FPNConvLSTM, self).__init__()
        self.pad_value = pad_value
        self.inconv = ConvBlock(
            nkernels=[input_dim] + inconv, norm="group", pad_value=pad_value
        )
        self.pyramid = PyramidBlock(
            input_dim=inconv[-1],
            n_channels=n_channels,
            n_levels=n_levels,
            pad_value=pad_value,
        )

        if mid_conv:
            dim = n_channels * n_levels // 2
            self.mid_conv = ConvBlock(
                nkernels=[self.pyramid.out_channels, dim],
                pad_value=pad_value,
                norm="group",
            )
        else:
            dim = self.pyramid.out_channels
            self.mid_conv = None

        self.convlstm = ConvLSTM(
            input_dim=dim,
            input_size=input_shape,
            hidden_dim=hidden_size,
            kernel_size=(3, 3),
            return_all_layers=False,
        )

        self.outconv = nn.Conv2d(
            in_channels=hidden_size, out_channels=num_classes, kernel_size=1
        )

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        pad_mask = pad_mask if pad_mask.any() else None

        out = self.inconv.smart_forward(input)
        out = self.pyramid.smart_forward(out)
        if self.mid_conv is not None:
            out = self.mid_conv.smart_forward(out)
        _, out = self.convlstm(out, pad_mask=pad_mask)
        out = out[0][1]
        out = self.outconv(out)

        return out


class TemporallySharedBlock(nn.Module):
    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class PyramidBlock(TemporallySharedBlock):
    def __init__(self, input_dim, n_levels=5, n_channels=64, pad_value=None):
        """
        Feature Pyramid Block. Performs atrous convolutions with different strides
        and concatenates the resulting feature maps along the channel dimension.
        Args:
            input_dim (int): Number of channels in the input images.
            n_levels (int): Number of levels.
            n_channels (int): Number of channels per level.
            pad_value (float): Padding value (temporal) used by the dataloader.
        """
        super(PyramidBlock, self).__init__(pad_value=pad_value)

        dilations = [2 ** i for i in range(n_levels - 1)]
        self.inconv = nn.Conv2d(input_dim, n_channels, kernel_size=3, padding=1)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=d,
                    padding=d,
                    padding_mode="reflect",
                )
                for d in dilations
            ]
        )

        self.out_channels = n_levels * n_channels

    def forward(self, input):
        out = self.inconv(input)
        global_avg_pool = out.view(*out.shape[:2], -1).max(dim=-1)[0]

        out = torch.cat([cv(out) for cv in self.convs], dim=1)

        h, w = out.shape[-2:]
        out = torch.cat(
            [
                out,
                global_avg_pool.unsqueeze(-1)
                .repeat(1, 1, h)
                .unsqueeze(-1)
                .repeat(1, 1, 1, w),
            ],
            dim=1,
        )

        return out


class ConvLayer(nn.Module):
    def __init__(self, nkernels, norm="batch", k=3, s=1, p=1, n_groups=4):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats, num_groups=n_groups
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode="reflect",
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(self, nkernels, pad_value=None, norm="batch"):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(nkernels=nkernels, norm=norm)

    def forward(self, input):
        return self.conv(input)
