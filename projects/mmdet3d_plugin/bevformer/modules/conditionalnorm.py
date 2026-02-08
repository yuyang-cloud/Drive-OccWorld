# ray normalization.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.models import HEADS
from mmcv.cnn import Linear, bias_init_with_prob
import math
import time
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from einops import rearrange, repeat

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def fourier_embedding(input, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal input embeddings.

    :param input: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(input, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=input.device)
        args = input[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding

class Fourier_Embed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return fourier_embedding(t, self.dim)
    
@HEADS.register_module()
class ConditionalNorm(BaseModule):
    """Ray marching adaptor for fine-tuning weights pre-trained by DriveGPT."""

    def __init__(self,
                 occ_flow='occ',
                 embed_dims=256,
                 num_pred_fcs=2,
                 pred_height=1,
                 num_cls=8,
                 act='exp',
                 sem_norm=False,
                 sem_gt_train=True,
                 ego_motion_ln=False,
                 obj_motion_ln=False,

                 viz_response=False,
                 init_cfg=None):

        super().__init__(init_cfg)

        self.occ_flow = occ_flow
        self.embed_dims = embed_dims
        self.num_pred_fcs = num_pred_fcs 
        self.viz_response = viz_response
        self.sem_norm = sem_norm
        self.sem_gt_train = sem_gt_train
        self.ego_motion_ln = ego_motion_ln
        self.obj_motion_ln = obj_motion_ln

        # Activation function should be:
        #  'exp' or 'sigmoid'
        self.act = act

        if self.sem_norm and self.occ_flow=='occ':
            # build up prob layer.
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs+1):
                sem_raymarching_branch.append(Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(Linear(self.embed_dims, num_cls * pred_height))
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)
            self.num_cls = num_cls
            self.pred_height = pred_height

            self.sem_param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)
            self.channel2height = Linear(self.embed_dims, self.embed_dims)

            nhidden = 32
            self.sem_mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_cls, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.sem_mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.pred_height, kernel_size=3, padding=1)
            self.sem_mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.pred_height, kernel_size=3, padding=1)

            # self.reset_parameters()
            nn.init.zeros_(self.sem_mlp_gamma.weight)
            nn.init.ones_(self.sem_mlp_gamma.bias)
            nn.init.zeros_(self.sem_mlp_beta.weight)
            nn.init.zeros_(self.sem_mlp_beta.bias)
        
        if self.ego_motion_ln:
            self.ego_param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 256
            self.ego_mlp_shared = nn.Sequential(
                nn.Linear(144, nhidden),
                nn.ReLU(),
            )
            self.ego_mlp_gamma = nn.Linear(nhidden, nhidden)
            self.ego_mlp_beta = nn.Linear(nhidden, nhidden)
            # self.reset_parameters()
            nn.init.zeros_(self.ego_mlp_gamma.weight)
            nn.init.zeros_(self.ego_mlp_beta.weight)
            nn.init.ones_(self.ego_mlp_gamma.bias)
            nn.init.zeros_(self.ego_mlp_beta.bias)
        
        if self.obj_motion_ln and self.occ_flow=='flow':
            self.fourier_embed = Fourier_Embed(32)
            
            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            self.pred_height = pred_height  # 16
            self.mlp_shared = nn.Sequential(
                nn.Linear(3*32, 64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dims // self.pred_height),
                nn.ReLU(),
            )
            self.mlp_gamma = nn.Linear(self.embed_dims // self.pred_height, self.embed_dims // self.pred_height)
            self.mlp_beta = nn.Linear(self.embed_dims // self.pred_height, self.embed_dims // self.pred_height)
            self.reset_parameters()

    
    def reset_parameters(self):
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.ones_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward_sem_norm(self,
                        embed,
                        sem_label=None,
                        **kwargs):
        """
        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
        """

        # 1. obtain unsupervised occupancy prediction.
        sem_pred = self.sem_raymarching_branch(embed)
        sem_pred = sem_pred.view(
            *sem_pred.shape[:-1], self.pred_height, self.num_cls)

        if not self.sem_gt_train or sem_label is None:
            sem_label = torch.argmax(sem_pred.detach(), dim=-1)
        sem_code = F.one_hot(sem_label.long(), num_classes=self.num_cls).float().permute(0,4,1,2,3).contiguous()

        # 2. generate parameter-free normalized activations
        embed = self.sem_param_free_norm(embed)
        embed = self.channel2height(embed).view(*embed.shape[:-1], self.pred_height, -1)
        embed = embed.permute(0,4,1,2,3)

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.sem_mlp_shared(sem_code)
        gamma = self.sem_mlp_gamma(actv)
        beta = self.sem_mlp_beta(actv)

        # apply scale and bias
        embed = (gamma * embed + beta).permute(0,2,3,4,1).flatten(3,4).contiguous()
        return embed, sem_pred.permute(0,4,1,2,3)

    def forward_ego_motion_ln(self,
                            embed,
                            future2history=None,
                            **kwargs):
        """
        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
            future2history: bs, 4, 4
        """
        # 1. memory_ego_motion
        memory_ego_motion = future2history[:, :3, :].flatten(-2).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)

        # 2. generate parameter-free normalized activations
        embed = self.ego_param_free_norm(embed)

        # 3. produce scaling and bias conditioned on semantic map
        actv = self.ego_mlp_shared(memory_ego_motion)
        gamma = self.ego_mlp_gamma(actv)
        beta = self.ego_mlp_beta(actv)

        # apply scale and bias
        embed = gamma * embed + beta
        return embed

    def forward_obj_motion_ln(self,
                            embed,
                            flow_3D=None,
                            occ_3D=None,
                            **kwargs):
        """
        Args:
            embed (Tensor): feature embedding after transformer layers.
                `(bs, bev_h, bev_w, embed_dims)`
            flow_3D: bs, 3, H, W, D
            occ_3D: bs, H, W, D
        """
        # 1. rearrange
        flow_3D = flow_3D.permute(0,2,3,4,1)
        b, h, w, d, dims = flow_3D.shape
        flow_3D = flow_3D * (occ_3D > 0).float().unsqueeze(-1)
        flow_3D = rearrange(flow_3D, "b h w d c -> (b h w d c)")

        # 2. fourier embed
        flow_3D = self.fourier_embed(flow_3D)
        flow_3D = rearrange(flow_3D, "(b h w d c) c2 -> b h w d (c c2)", b=b, h=h, w=w, d=d, c=dims, c2=self.fourier_embed.dim)

        # 3. generate parameter-free normalized activations
        embed = self.param_free_norm(embed)

        # 4. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(flow_3D)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        embed = gamma * embed.view(b, h, w, d, -1) + beta
        embed = embed.view(b, h, w, -1)
        return embed
    
    def forward(self, embeds, cond_norm_dict, flow_3D=None, occ_3D=None):
        """Forward Function of Rendering.

        Args:
            embeds (Tensor): BEV feature embeddings.
                `(bs, F, bev_h, bev_w, embed_dims)`
        """
        bs, num_frames, bev_h, bev_w, embed_dim = embeds.shape
        occ_gts, future2history = cond_norm_dict['occ_gts'], cond_norm_dict['future2history']

        if self.sem_norm and self.occ_flow=='occ':
            sem_embeds = []
            for i in range(num_frames):
                if occ_gts is not None:
                    sem_embed, bev_sem_pred = self.forward_sem_norm(embeds[:, i, ...], occ_gts[:, i, ...])
                else:
                    sem_embed, bev_sem_pred = self.forward_sem_norm(embeds[:, i, ...])
                sem_embeds.append(sem_embed)
            sem_embeds = torch.stack(sem_embeds, dim=1)

        if self.ego_motion_ln and future2history is not None:
            ego_motion_embeds = []
            for i in range(num_frames):
                motion_embed = self.forward_ego_motion_ln(embeds[:, i, ...], future2history[:, i, ...])
                ego_motion_embeds.append(motion_embed)
            ego_motion_embeds = torch.stack(ego_motion_embeds, dim=1)

        if self.obj_motion_ln and flow_3D is not None:
            obj_motion_embeds = []
            for i in range(num_frames):
                motion_embed = self.forward_obj_motion_ln(embeds[:, i, ...], flow_3D[:, i, ...], occ_3D[:, i, ...])
                obj_motion_embeds.append(motion_embed)
            obj_motion_embeds = torch.stack(obj_motion_embeds, dim=1)


        if self.sem_norm and self.occ_flow=='occ' and self.ego_motion_ln and future2history is not None:
            embeds = sem_embeds + ego_motion_embeds
        elif self.sem_norm and self.occ_flow=='occ':
            embeds = sem_embeds
        elif self.ego_motion_ln and future2history is not None:
            embeds = ego_motion_embeds


        embeds = embeds.transpose(2, 3).contiguous() # NOTE: query first_H, then_W
        out_dict = {
            'bev_embed': embeds.view(bs, num_frames, -1, embed_dim),
            'bev_sem_pred': bev_sem_pred if self.sem_norm and self.occ_flow=='occ' and self.training else None,
        }
        return out_dict
