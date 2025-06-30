import torch
import torch.nn as nn
import torch.nn.functional as F
NORM = nn.InstanceNorm3d
ACT = nn.ReLU(inplace=True)


class SyMNet(torch.nn.Module):
    def __init__(self):
        super(SyMNet, self).__init__()
        self.unet = Unet3d(n_in=4, n_out=6, n_ch=32, act_module=torch.nn.LeakyReLU())

    def forward(self, x, y=None):
        out = x if y is None else torch.cat([x, y], 1)
        out = F.pad(out, (7, 8, 3, 4, 7, 8))
        out = self.unet(out)
        out = out[:, :, 7:-8, 3:-4, 7:-8]
        out = .1 * torch.tanh(out)
        return (out[:, :3], out[:, 3:]) if y is None else (out[:, :3], out[:, 3:], x, y)


class TwoInputsUnet3d(torch.nn.Module):
    def __init__(self, unet, act):
        super(TwoInputsUnet3d, self).__init__()
        self.unet = unet
        self.act = act

    def forward(self, x1, x2=None):
        x = x1 if x2 is None else torch.cat([x1, x2[:, None]], 1)
        return self.act(self.unet(x))


class StepActivation(torch.nn.Module):
    def __init__(self, factor=10., half_margin=.2):
        super(StepActivation, self).__init__()
        self.factor = torch.nn.Parameter(torch.tensor([float(factor)]))
        self.half_margin = torch.nn.Parameter(torch.tensor([float(half_margin)]))

    def forward(self, x):
        return (torch.sigmoid(self.factor * x) + \
                .5 * torch.sigmoid(self.factor * x - (1 - self.half_margin) * self.factor) + \
                .5 * torch.sigmoid(self.factor * x - (1 + self.half_margin) * self.factor) + \
                .5 * torch.sigmoid(self.factor * x - (2 - self.half_margin) * self.factor) + \
                .5 * torch.sigmoid(self.factor * x - (2 + self.half_margin) * self.factor))


class Unet3d(nn.Module):
    def __init__(self, n_in=1, n_out=2, n_ch=8, depth=4, kernel=3, norm_func=NORM, act_module=ACT, p_dropout=.0,
                 n_chs=None, start_pad=None, final_pad=0):
        super(Unet3d, self).__init__()
        self.depth = depth
        n_chs = [(n_ch * 2 ** i, n_ch * 2 ** (i + 1)) for i in range(depth)] if n_chs is None else n_chs
        self.inp = DoubleConv(n_in, n_ch, kernel, norm_func, act_module, p_dropout, pad=start_pad)
        for i, (ch_in, ch_out) in enumerate(n_chs):
            setattr(self, f'down{i}', Down(ch_in, ch_out, kernel, norm_func, act_module, p_dropout))
        for i, (ch_in, ch_out) in enumerate(n_chs[::-1]):
            setattr(self, f'up{i}', Up(ch_out, ch_in, kernel, norm_func, act_module, p_dropout))
        self.out = nn.Conv3d(n_ch, n_out, 1, padding=final_pad)

    def forward(self, x):
        xs = []
        x = self.inp(x)
        for i in range(self.depth):
            xs.append(x)
            block = getattr(self, f'down{i}')
            x = block(x)
        for i in range(self.depth):
            block = getattr(self, f'up{i}')
            x = block(x, xs.pop())
        return self.out(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, norm_func, act_module, p_dropout):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, kernel, norm_func, act_module, p_dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, norm_func, act_module, p_dropout):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, kernel, norm_func, act_module, p_dropout)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, norm_func, act_module, p_dropout, pad=None):
        super(DoubleConv, self).__init__()
        conv1 = [nn.Conv3d(in_ch, out_ch, kernel, padding=kernel // 2 if pad is None else pad), norm_func(out_ch), act_module]
        if p_dropout > 0:
            conv1.append(nn.Dropout3d(p_dropout))
        conv2 = [nn.Conv3d(out_ch, out_ch, kernel, padding=kernel // 2), norm_func(out_ch), act_module]
        self.double_conv = nn.Sequential(*conv1, *conv2)

    def forward(self, x):
        return self.double_conv(x)
