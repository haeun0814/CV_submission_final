import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SeparableConv2d(nn.Module):
    """3×3 Depthwise + 1×1 Pointwise 분리 합성곱"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.act(self.bn(x))
        x = self.pw(x)
        return x

class MultiDilationSeparableConv2d(nn.Module):
    """동일 입력을 dilation 1 / dilation d 로 나눠 처리 후 합산"""
    def __init__(self, in_ch, out_ch, k=3, d=2, bias=True):
        super().__init__()
        p1 = k // 2
        p2 = p1 + (d - 1) * (k - 1) // 2
        self.dw1 = nn.Conv2d(in_ch, in_ch, k, 1, p1, groups=in_ch, bias=False)
        self.dw2 = nn.Conv2d(in_ch, in_ch, k, 1, p2, d, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.act(self.bn1(self.dw1(x)))
        x2 = self.act(self.bn2(self.dw2(x)))
        return self.pw(x1 + x2)

class SEModule(nn.Module):
    """Squeeze-and-Excitation 모듈"""
    def __init__(self, ch, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn = self.net(self.pool(x))
        return x * attn

class MicroDownsampleModuleV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_pool = in_ch < out_ch
        conv_out = out_ch if not self.use_pool else out_ch - in_ch
        self.conv = SeparableConv2d(in_ch, conv_out, k=3, s=2, p=1)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.conv(x)
        if self.use_pool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2)], dim=1)
        return self.act(self.bn(y))

class MicroResidualConvModule(nn.Module):
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        self.conv = SeparableConv2d(ch, ch, 3, 1, dil, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)
        self.act = nn.SiLU(inplace=True)
        self.se = SEModule(ch)
    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        y = self.se(y)
        return self.act(x + y)

class MicroResidualMultiDilationConvModule(nn.Module):
    def __init__(self, ch, dil=2, drop=0.):
        super().__init__()
        self.conv = MultiDilationSeparableConv2d(ch, ch, 3, dil, False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3)
        self.drop = nn.Dropout2d(drop)
        self.act = nn.SiLU(inplace=True)
        self.se = SEModule(ch)
    def forward(self, x):
        y = self.drop(self.bn(self.conv(x)))
        y = self.se(y)
        return self.act(x + y)

class MicroUpsampleModuleV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = SeparableConv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return self.act(self.bn(x))

class GradientFeatureModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=torch.float32).expand(ch, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.refine = SeparableConv2d(ch * 2, ch, 1, 1, 0, 1, False)
        self.bn     = nn.BatchNorm2d(ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.size(1))
        g  = torch.cat([gx, gy], dim=1)
        return self.act(self.bn(self.refine(g)))

class MicroNetV5EncoderV2(nn.Module):
    def __init__(self, in_ch: int, ch: tuple = (10, 20, 26), rates: tuple = (1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3 = ch
        self.down1 = MicroDownsampleModuleV2(in_ch, c1)
        self.grad  = GradientFeatureModule(c1)
        self.down2 = MicroDownsampleModuleV2(c1, c2)
        self.mid   = nn.Sequential(MicroResidualConvModule(c2, 1, 0.0), MicroResidualConvModule(c2, 1, 0.0))
        self.down3 = MicroDownsampleModuleV2(c2, c3)
        self.ctx   = nn.Sequential(*[MicroResidualMultiDilationConvModule(c3, d, 0.1) for d in rates])
    def forward(self, x):
        d1 = self.down1(x)
        d1 = d1 + self.grad(d1)
        d2 = self.mid(self.down2(d1))
        d3 = self.down3(d2)
        out = self.ctx(d3)
        return out, d2

class GhostSeparableConv2d(nn.Module):
    """👻 GhostNet 아이디어를 적용한 분리 합성곱 (파라미터 감소)"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bias=True, ratio=2):
        super().__init__()
        self.out_ch = out_ch
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
        primary_ch = math.ceil(out_ch / ratio)
        cheap_ch = primary_ch * (ratio - 1)
        self.pw_primary = nn.Conv2d(in_ch, primary_ch, 1, bias=bias)
        self.pw_cheap = nn.Conv2d(primary_ch, cheap_ch, 3, 1, 1, groups=primary_ch, bias=bias)
        self.bn_pw = nn.BatchNorm2d(primary_ch + cheap_ch, eps=1e-3)

    def forward(self, x):
        x = self.act(self.bn_dw(self.dw(x)))
        x_primary = self.pw_primary(x)
        x_cheap = self.pw_cheap(x_primary)
        x_out = torch.cat([x_primary, x_cheap], dim=1)
        return self.bn_pw(x_out)[:, :self.out_ch, :, :]

class MicroResidualGhostConvModule(nn.Module):
    """👻 GhostSeparableConv2d를 사용하는 Residual 블록"""
    def __init__(self, ch, dil=1, drop=0.):
        super().__init__()
        self.conv = GhostSeparableConv2d(ch, ch, 3, 1, dil, dil, False)
        self.drop = nn.Dropout2d(drop)
        self.act = nn.SiLU(inplace=True)
        self.se = SEModule(ch)
    def forward(self, x):
        y = self.drop(self.conv(x))
        y = self.se(y)
        return self.act(x + y)

class MicroASPPModule(nn.Module):
    """✨ [수정됨] 경량화된 Atrous Spatial Pyramid Pooling 모듈 (성능 향상)"""
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        # 1x1 Conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        # Atrous Separable Conv
        self.aspp_convs = nn.ModuleList()
        for rate in rates:
            self.aspp_convs.append(SeparableConv2d(in_ch, out_ch, 3, 1, p=rate, d=rate, bias=False))

        # --- 🚨 오류 수정 부분 ---
        # Global Average Pooling에서 BatchNorm을 분리
        self.pool_proc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.SiLU(inplace=True)
        )
        # BatchNorm은 업샘플링 이후에 적용
        self.pool_bn = nn.BatchNorm2d(out_ch)
        # ------------------------
        
        # 최종 융합 레이어
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        res1x1 = self.conv1x1(x)
        res_aspp = [conv(x) for conv in self.aspp_convs]
        
        # --- 🚨 오류 수정 부분 ---
        # Pool -> Conv -> Upsample -> BN 순서로 실행
        res_pool = self.pool_proc(x)
        res_pool = F.interpolate(res_pool, size=size, mode='bilinear', align_corners=False)
        res_pool = self.pool_bn(res_pool)
        # ------------------------
        
        out = torch.cat([res1x1] + res_aspp + [res_pool], dim=1)
        return self.project(out)

class MicroNetV5EncoderV2_ASPP(nn.Module):
    """✨ 기존 Encoder의 ctx를 ASPP로 교체한 버전"""
    def __init__(self, in_ch: int, ch: tuple = (10, 20, 26), rates: tuple = (6, 12, 18)):
        super().__init__()
        c1, c2, c3 = ch
        self.down1 = MicroDownsampleModuleV2(in_ch, c1)
        self.grad  = GradientFeatureModule(c1)
        self.down2 = MicroDownsampleModuleV2(c1, c2)
        self.mid   = nn.Sequential(MicroResidualConvModule(c2, 1, 0.0), MicroResidualConvModule(c2, 1, 0.0))
        self.down3 = MicroDownsampleModuleV2(c2, c3)
        self.ctx   = MicroASPPModule(c3, c3, rates)
    def forward(self, x):
        d1 = self.down1(x)
        d1 = d1 + self.grad(d1)
        d2 = self.mid(self.down2(d1))
        d3 = self.down3(d2)
        out = self.ctx(d3)
        return out, d2


class submission_20223292(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        interpolate = True
        self.interpolate = interpolate
        ch = (8, 14, 18)
        c1, c2, c3 = ch
        self.encoder = MicroNetV5EncoderV2_ASPP(in_channels, ch=ch, rates=(6,12,18))
        self.aux_ds  = MicroDownsampleModuleV2(in_channels, c1)
        self.aux_ref = MicroResidualConvModule(c1, 1, 0.0)
        self.up1     = MicroUpsampleModuleV2(c3, c2)
        self.up_mid  = nn.Sequential(MicroResidualConvModule(c2, 1, 0.0), MicroResidualConvModule(c2, 1, 0.0))
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fusion_conv = SeparableConv2d(c1 + c2, c2, k=3, s=1, p=1)
        self.head_conv = nn.Conv2d(c2, num_classes, 1)
    def forward(self, x):
        input_size = x.shape[2:]; enc, skip = self.encoder(x); aux = self.aux_ref(self.aux_ds(x)); y = self.up1(enc)
        if y.size(2) != skip.size(2) or y.size(3) != skip.size(3): y = F.interpolate(y, size=skip.shape[2:], mode='bilinear', align_corners=False)
        y = y + skip; y = self.up_mid(y); y = self.final_up(y); y = torch.cat([y, aux], dim=1); y = self.fusion_conv(y); out = self.head_conv(y)
        if self.interpolate and out.shape[2:] != input_size: out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out