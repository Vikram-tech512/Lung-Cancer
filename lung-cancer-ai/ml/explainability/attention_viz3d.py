"""
backend/explainability/attention_viz3d.py
=========================================
3D Vision Transformer Attention Rollout and Map Visualizer (Abnar & Zuidema, 2020).
Projects multi-layer self-attention across 3D patch tokens back to original voxel coordinate space.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class AttentionRollout3D:
    def __init__(self, discard_ratio: float = 0.9, head_fusion: str = "mean"):
        """
        discard_ratio: fraction of lowest attention weights to zero out (reduces noise)
        head_fusion: 'mean', 'max', or 'min' across multi-head attention
        """
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def compute_rollout(self, attention_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Takes list of attention maps from each transformer layer:
        Each matrix shape: (batch_size, num_heads, num_tokens, num_tokens)
        Returns: aggregated attention matrix (num_tokens, num_tokens)
        """
        result = torch.eye(attention_matrices[0].size(-1)).to(attention_matrices[0].device)

        with torch.no_grad():
            for attn in attention_matrices:
                if self.head_fusion == "mean":
                    attn_fused = attn.mean(dim=1)
                elif self.head_fusion == "max":
                    attn_fused = attn.max(dim=1)[0]
                else:
                    attn_fused = attn.min(dim=1)[0]

                # Discard lowest attention values per row
                flat = attn_fused.view(attn_fused.size(0), -1)
                k = int(flat.size(-1) * self.discard_ratio)
                if k > 0:
                    val, _ = torch.kthvalue(flat, k, dim=-1, keepdim=True)
                    attn_fused = torch.where(attn_fused < val.unsqueeze(-1), torch.zeros_like(attn_fused), attn_fused)

                # Add identity matrix to account for residual token paths
                I = torch.eye(attn_fused.size(-1)).to(attn_fused.device)
                a = (attn_fused + I) / 2.0
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        return result[0]  # First batch item

    def project_to_volume(
        self,
        token_attentions: torch.Tensor,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        target_shape: Tuple[int, int, int] = (64, 64, 64)
    ) -> Dict[str, Any]:
        """
        Maps token-level attention back to 3D voxel grid.
        grid_shape: (D_tokens, H_tokens, W_tokens)
        target_shape: (D_orig, H_orig, W_orig)
        """
        # CLS token attention to spatial tokens (or mean across all tokens if no CLS)
        if token_attentions.ndim == 2:
            spatial_attn = token_attentions[0, 1:] if token_attentions.shape[0] > np.prod(grid_shape) else token_attentions.mean(dim=0)
        else:
            spatial_attn = token_attentions

        num_tokens = int(np.prod(grid_shape))
        spatial_vec = spatial_attn[:num_tokens].cpu().numpy()
        grid_3d = spatial_vec.reshape(grid_shape)

        # Upsample grid to target shape via trilinear interpolation
        import scipy.ndimage
        zoom_factors = [t / g for t, g in zip(target_shape, grid_shape)]
        upsampled = scipy.ndimage.zoom(grid_3d, zoom_factors, order=1)
        norm_map = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)

        mip_axial = np.max(norm_map, axis=0)
        mip_coronal = np.max(norm_map, axis=1)
        mip_sagittal = np.max(norm_map, axis=2)

        return {
            "attention_map_3d": norm_map,
            "mip_axial": mip_axial,
            "mip_coronal": mip_coronal,
            "mip_sagittal": mip_sagittal,
            "method": "3D Swin Attention Rollout",
            "disclaimer": "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE. Research Use Only."
        }
