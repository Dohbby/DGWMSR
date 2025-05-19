import torch
import torch.nn.functional as F

from .warplayer import warp
from basicsr.utils.registry import ARCH_REGISTRY

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from .cfat import CFAT
from .hi import HiT_SRF
# from .DRSformer_arch import DRSformer
# from .DAT_MRI import  DAT_MRI

from .lu import SRGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                                 bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 解码器
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, img0, img1, warped_img0, warped_img1):
        x = torch.cat((img0, img1, warped_img0, warped_img1), 1)
        # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        # 解码器（带跳跃连接）
        d1 = self.decoder1(torch.cat((e2, e1), 1))
        # 最终输出
        out = self.final_conv(d1)
        out = out + warped_img0 * 0.5 + warped_img1 * 0.5
        return torch.sigmoid(out)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h + pad_h, w + pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h // 2),
            slice(pad_h // 2, h + pad_h // 2),
            slice(h + pad_h // 2, None),
        )
        w_slices = (
            slice(0, pad_w // 2),
            slice(pad_w // 2, w + pad_w // 2),
            slice(w + pad_w // 2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2: pad_h // 2 + h, pad_w // 2: pad_w // 2 + w, :].contiguous()
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(2, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2, cor, H, W, mask=None):
        B, N, C = x1.shape
        B, N, C_c = cor.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]  # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse - cor_embed_)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, motion


class MotionFormerBlock(nn.Module):
    def __init__(self, dim, motion_dim, num_heads, window_size=0, shift_size=0, mlp_ratio=4., bidirectional=True,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttention(
            dim,
            motion_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # # 在初始化时注册默认的 attn_mask 和 HW
        # self.register_buffer("attn_mask", torch.zeros(16, 256, 256))  # 初始化为 None，允许动态更新
        # self.register_buffer("HW",torch.tensor([0]))  # 初始化为 None，允许动态更新

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, cor, H, W, B):
        x = x.view(2 * B, H, W, -1)
        x_pad, mask = pad_if_needed(x, x.size(), self.window_size)
        cor_pad, _ = pad_if_needed(cor, cor.size(), self.window_size)

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            cor_pad = torch.roll(cor_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

            if hasattr(self, 'HW') and self.HW.item() == H_p * W_p:
                shift_mask = self.attn_mask.to(x_pad.device)
            else:
                shift_mask = torch.zeros((1, H_p, W_p, 1), device=x_pad.device)  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)
                shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                shift_mask = shift_mask.masked_fill(shift_mask != 0,
                                                    float(-100.0)).masked_fill(shift_mask == 0,
                                                                               float(0.0))
                if mask is not None:
                    shift_mask = shift_mask.masked_fill(mask != 0,
                                                        float(-100.0))
                # self.register_buffer("attn_mask", shift_mask)
                # hw_tensor = torch.tensor([H_p * W_p], device=x_pad.device)
                # # self.register_buffer("HW", torch.Tensor([H_p * W_p]))
                # self.register_buffer("HW",hw_tensor)
        else:
            shift_mask = mask

        if shift_mask is not None:
            shift_mask = shift_mask.to(x_pad.device)

        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size)
        cor_win = window_partition(cor_pad, self.window_size)

        nwB = x_win.shape[0]
        x_norm = self.norm1(x_win)

        x_reverse = torch.cat([x_norm[nwB // 2:], x_norm[:nwB // 2]])
        x_appearence, x_motion = self.attn(x_norm, x_reverse, cor_win, H, W, shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm
        x_back_win = window_reverse(x_back, self.window_size, Hw, Ww)
        x_motion = window_reverse(x_motion, self.window_size, Hw, Ww)

        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x_motion = torch.roll(x_motion, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x.size(), self.window_size).view(2 * B, H * W, -1)
        x_motion = depad_if_needed(x_motion, cor.size(), self.window_size).view(2 * B, H * W, -1)

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, x_motion


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2, act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class CrossScalePatchEmbed(nn.Module):
    def __init__(self, in_dims=[16, 32, 64], embed_dim=768):
        super().__init__()
        base_dim = in_dims[0]

        layers = []
        for i in range(len(in_dims)):
            layers.append(nn.Conv2d(in_dims[-1 - i], base_dim, 3, 1, 1, 1))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs)):
            ys.append(self.layers[k](xs[-1 - i]))
            k += 1
        x = self.proj(torch.cat(ys, 1))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MotionFormer(nn.Module):
    def __init__(self, in_chans=1,
                 embed_dims=[64, 128, 256],
                 motion_dims=[0, 64, 128],
                 num_heads=[8, 16],
                 mlp_ratios=[4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[4,2,2],
                 window_sizes=[16, 16]):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans, embed_dims[i], depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i - 1], embed_dims[i], 3, 2, 1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i], embed_dims[i], depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = CrossScalePatchEmbed(embed_dims[:i],
                                                           embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=1,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    block = nn.ModuleList([MotionFormerBlock(
                        dim=embed_dims[i], motion_dim=motion_dims[i], num_heads=num_heads[i - self.conv_stages],
                        window_size=window_sizes[i - self.conv_stages],
                        shift_size=0 if (j % 2) == 0 else window_sizes[i - self.conv_stages] // 2,
                        mlp_ratio=mlp_ratios[i - self.conv_stages], qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer)
                        for j in range(depths[i])])

                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0]
        x = torch.cat([x1, x2], 0)
        motion_features = []
        appearence_features = []
        xs = []
        for i in range(self.num_stages):
            motion_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}", None)
            block = getattr(self, f"block{i + 1}", None)
            norm = getattr(self, f"norm{i + 1}", None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
                xs.append(x)
            else:
                if i == self.conv_stages:
                    x, H, W = patch_embed(xs)
                else:
                    x, H, W = patch_embed(x)
                cor = self.get_cor((x.shape[0], H, W), x.device)
                for blk in block:
                    x, x_motion = blk(x, cor, H, W, B)
                    motion_features[i].append(x_motion.reshape(2 * B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                x = norm(x)
                x = x.reshape(2 * B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                motion_features[i] = torch.cat(motion_features[i], 1)
            appearence_features.append(x)
        return appearence_features, motion_features


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        # self.upsample = nn.Sequential(nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
            conv(in_planes * 2 + in_else, c),
            conv(c, c),
            conv(c, 5),
        )

    def forward(self, motion_feature, x, flow):  # /16 /8 /4
        # motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor=1, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor=1, mode="bilinear", align_corners=False)
            x = torch.cat((x, flow), 1)
        #
        # print('x:',x.shape)
        # print('motion_feature:',motion_feature.shape)
        # print('3333333',torch.cat([motion_feature, x], 1).shape)

        x = self.conv(torch.cat([motion_feature, x], 1))
        # print('x:',x.shape)
        if self.scale != 4:
            x = F.interpolate(x, scale_factor=1, mode="bilinear", align_corners=False)
            flow = x[:, :4] * 1
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask


@ARCH_REGISTRY.register()
class MultiScaleFlow(nn.Module):
    def __init__(self, motion_dims=[0, 64, 128],
                 depths=[4, 2,2],
                 embed_dims=[64, 128, 256],
                 scales=[4, 8, 16],
                 hidden_dims=[256, 256]):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = 2
        self.feature_bone = MotionFormer()
        self.depths = depths
        self.embed_dims = embed_dims
        self.block = nn.ModuleList(
            [Head(motion_dims[-1 - i] * depths[-1 - i] + embed_dims[-1 - i],
                  scales[-1 - i],
                  hidden_dims[-1 - i],
                  2 if i == 0 else 9)
             for i in range(self.flow_num_stage)])
        self.unet = Unet()
        # self.cfat=CFAT()
        self.hi=HiT_SRF()
        # self.dsr=DRSformer()
        # self.dat=DAT_MRI()
        # self.lunet = SRGAN()

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=1, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False)
        return y0, y1

    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, img0, img1,x):
        timestep = 0.5
        # img0, img1 = x[:, :3], x[:, 3:6]
        B = img0.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # appearence_features & motion_features
        af, mf = self.feature_bone(img0, img1)

        # af = [tensor * 0 for tensor in af]
        # mf = [tensor * 0 for tensor in mf]

        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1 - i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                flow_d, mask_d = self.block[i](
                    torch.cat([t * mf[-1 - i][:B], (1 - timestep) * mf[-1 - i][B:], af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i](
                    torch.cat([t * mf[-1 - i][:B], (1 - t) * mf[-1 - i][B:], af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1), 1), None)
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))

        # c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1)
        # res = tmp[:, :3]  - 1
        pred1 = torch.clamp(merged[-1] + tmp, 0, 1)
        # print("1111:::",pred1.shape)
        # pred2=self.cfat(x)
        pred2=self.hi(x)
        # print(pred2.shape)
        # pred2 = self.lunet(x)
        pred = pred2 + pred1
        return pred


