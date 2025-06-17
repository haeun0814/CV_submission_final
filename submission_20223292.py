import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x = self.dw(x)
        x = F.relu(self.bn(x))
        x = self.pw(x)
        return x

class MultiDilationSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=2, bias=True): 
        super().__init__()
        p1 = p
        p2 = p + (d - 1) * (k - 1) // 2

        self.dw1 = nn.Conv2d(in_ch, in_ch, k, s, p1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3)

        self.dw2 = nn.Conv2d(in_ch, in_ch, k, s, p2, d, groups=in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch, eps=1e-3)
        
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.dw1(x)))
        x2 = F.relu(self.bn2(self.dw2(x)))
        out = x1 + x2
        return self.pw(out)

class MicroDownsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_pool = in_ch < out_ch
        conv_out = out_ch if not self.use_pool else out_ch - in_ch

        self.conv = nn.Conv2d(in_ch, conv_out, 3, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2)], dim=1)
        return F.relu(self.bn(y))

class MicroResidualConvModule(nn.Module):
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, False) 
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(self.bn(y))
        y = self.drop(y)
        return F.relu(x + y)

class MicroResidualMultiDilationConvModule(nn.Module):
    def __init__(self, ch, dil=2, drop=0.): 
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, 1, 1, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        y = self.conv(x)
        y = F.relu(self.bn(y))
        y = self.drop(y)
        return F.relu(x + y)

class MicroUpsampleModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, output_padding=1, bias=False)
        self.bn     = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)))


class GradientFeatureModule(nn.Module):
    def __init__(self, in_ch, out_ch_refine):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(in_ch, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=torch.float32).expand(in_ch, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.refine = SeparableConv2d(in_ch * 2, out_ch_refine, 1, 1, 0, 1, False)
        self.bn     = nn.BatchNorm2d(out_ch_refine, eps=1e-3)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        g  = torch.cat([gx, gy], dim=1) # (Batch, in_ch * 2, H, W)
        return F.relu(self.bn(self.refine(g))) # (Batch, out_ch_refine, H, W)

class MicroNetV5Encoder(nn.Module):

    def __init__(self, in_ch: int, ch: tuple = (6, 12, 18), rates: tuple = (1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3 = ch 
        self.grad  = GradientFeatureModule(in_ch, c1)

        self.down1 = MicroDownsampleModule(in_ch, c1) 
        
        self.down2 = MicroDownsampleModule(c1, c2)
        self.mid   = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )

        self.down3 = MicroDownsampleModule(c2, c3)
        self.ctx   = nn.Sequential(*[
            MicroResidualMultiDilationConvModule(c3, d, 0.1) for d in rates
        ])

    def forward(self, x):
        d1 = self.down1(x) 
        g_feat = self.grad(x) 
        
        if g_feat.shape[2:] != d1.shape[2:]:
            g_feat = F.avg_pool2d(g_feat, kernel_size=2, stride=2) 
        d1_enhanced = d1 + g_feat 
        d2 = self.mid(self.down2(d1_enhanced)) 
        d3 = self.down3(d2)
        out = self.ctx(d3) 

        return out, d2 # enc, skip

class submission_20223292(nn.Module):

    def __init__(self, in_ch: int, num_classes: int, ch: tuple = (6, 12, 18), interpolate: bool = True):
        super().__init__()
        self.interpolate = interpolate
        c1, c2, c3 = ch 

        self.encoder = MicroNetV5Encoder(in_ch, ch=ch)

        self.aux_ds  = MicroDownsampleModule(in_ch, c2)
        self.aux_ref = MicroResidualConvModule(c2, 1, 0.0) 
        self.up1     = MicroUpsampleModule(c3, c2) 
        self.up_mid  = nn.Sequential(
            MicroResidualConvModule(c2, 1, 0.0),
            MicroResidualConvModule(c2, 1, 0.0)
        )
        self.head    = nn.ConvTranspose2d(c2, num_classes, 3, 2, 1, output_padding=1)

    def forward(self, x):
        aux_raw = self.aux_ds(x)
        aux     = self.aux_ref(aux_raw) 

        enc, skip = self.encoder(x) 
        y = self.up1(enc)
        if y.shape[2:] == skip.shape[2:] and y.shape[1] == skip.shape[1]:
            y = y + skip
        
        aux_pooled = F.avg_pool2d(aux, kernel_size=2, stride=2)
        if y.shape[2:] == aux_pooled.shape[2:] and y.shape[1] == aux_pooled.shape[1]:
            y = y + aux_pooled

        y   = self.up_mid(y) 
        out = self.head(y) 

        if self.interpolate:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return out