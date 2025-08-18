import math
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image


class ViTAttentionMap:
    def __init__(self, model):
        self.model = model
        last_block = self.model.blocks[-1]
        self.attn_obj = last_block.attn
        self.attn_obj.fused_attn = False  # Disable fused_attn to compute explicit attention
        self.original_forward = self.attn_obj.forward
        self.attn_obj.forward = self.my_forward_wrapper(self.attn_obj)
        self.grid_size = int(math.sqrt(self.model.patch_embed.num_patches))

    def my_forward_wrapper(self, attn_obj):
        def my_forward(x, attn_mask=None):
            B, N, C = x.shape
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, attn_obj.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Handle qk_norm if enabled
            q = attn_obj.q_norm(q.reshape(B * attn_obj.num_heads, N, attn_obj.head_dim)).reshape(B, attn_obj.num_heads, N, attn_obj.head_dim)
            k = attn_obj.k_norm(k.reshape(B * attn_obj.num_heads, N, attn_obj.head_dim)).reshape(B, attn_obj.num_heads, N, attn_obj.head_dim)

            q = q * attn_obj.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)

            # Store attention map after softmax, before drop
            attn_obj.attn_map = attn
            attn_obj.cls_attn_map = attn[:, :, 0, 1:]  # [B, heads, num_patches]

            attn = attn_obj.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        return my_forward

    def __call__(self, input_tensor):
        _ = self.model(input_tensor)

        # Get cls_attn_map [B, heads, num_patches]
        cls_attn = self.attn_obj.cls_attn_map

        # Average over heads
        cls_attn = cls_attn.mean(dim=1)[0]  # [num_patches]

        # Reshape to grid
        attn_map = cls_attn.view(self.grid_size, self.grid_size)

        # Interpolate to 224x224
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        return attn_map.cpu().numpy()


def get_overlaid_image(original_img, heatmap, alpha=0.4):
    orig_np = np.array(original_img) / 255.0

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap_img = jet_heatmap_img.resize(original_img.size, Image.BILINEAR)
    jet_heatmap_np = np.array(jet_heatmap_img) / 255.0

    superimposed_np = jet_heatmap_np * alpha + orig_np * (1 - alpha)
    superimposed_img = Image.fromarray(np.uint8(superimposed_np * 255))

    return superimposed_img

