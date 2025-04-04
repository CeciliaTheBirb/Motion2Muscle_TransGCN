from torch import nn
import torch
#modified attention to attend to MASKs

class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C)
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, J, C]
            mask: [B, T] (optional) — only used for 'temporal' mode
        """
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, H, T, J, C)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode == 'temporal':
            x = self.forward_temporal(q, k, v, mask)
        elif self.mode == 'spatial':
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, mask=None):
        B, H, T, J, C = q.shape
        # Transpose to bring T to the appropriate dimensions
        qt = q.transpose(2, 3)  # [B, H, J, T, C]
        kt = k.transpose(2, 3)  # [B, H, J, T, C]
        vt = v.transpose(2, 3)  # [B, H, J, T, C]

        # Compute raw attention scores: [B, H, J, T, T]
        attn = (qt @ kt.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expand mask from [B, T] to [B, 1, 1, 1, T] so that it masks the key (last) dimension
            mask_expanded = mask[:, None, None, None, :].to(attn.device)  # [B, 1, 1, 1, T]
            # Set attention scores for padded positions to -inf so that softmax will zero them out
            attn = attn.masked_fill(~mask_expanded, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # [B, H, J, T, C]
        # Rearrange back to [B, T, J, (H * C)]
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x

