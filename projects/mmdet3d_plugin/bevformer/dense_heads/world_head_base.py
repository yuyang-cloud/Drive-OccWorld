import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import xavier_init, constant_init
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from torch.nn.init import normal_
from ..utils import e2e_predictor_utils
from mmdet3d.models import builder
from einops import rearrange, repeat
from ..modules.conditionalnorm import Fourier_Embed as Fourier_Embed_v1
import math

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
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * self.embed_dim, 256),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            xavier_init(self.mlp, distribution='uniform', bias=0.)
        except:
            pass

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, c = t.shape
        # fourier embed
        t = rearrange(t, "b c -> (b c)")
        t = fourier_embedding(t, self.embed_dim)
        t = rearrange(t, "(b c) c2 -> b (c c2)", b=b, c=c, c2=self.embed_dim)
        # mlp
        t = self.mlp(t)
        return t

@HEADS.register_module()
class WorldHeadTemplate(BaseModule):

    def __init__(self,
                 *args,
                 num_classes,
                 # Architecture.
                 prev_render_neck=None,
                 transformer=None,
                 num_pred_fcs=2,
                 num_pred_height=1,

                 # Memory Queue configurations.
                 memory_queue_len=1,
                 sem_norm=False,

                 # Embedding configuration.
                 use_can_bus=False,
                 can_bus_norm=True,
                 can_bus_dims=(0, 1, 2, 17),
                 use_plan_traj=True,
                 use_command=False,
                 use_vel_steering=False,
                 use_vel=False,
                 use_steering=False,
                 condition_ca_add='add',
                 use_fourier=True,

                 # target BEV configurations.
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,

                 # loss functions.
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # evaluation configuration.
                 eval_within_grid=False,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        # action_condition
        self.condition_ca_add = condition_ca_add
        self.use_fourier = use_fourier

        # memory queue
        self.memory_queue_len = memory_queue_len
        self.sem_norm = sem_norm
        # fourier_embed
        self.fourier_nhidden = 64
        # Embedding configurations.
        self.can_bus_norm = can_bus_norm
        self.use_can_bus = use_can_bus
        self.can_bus_dims = can_bus_dims  # (delta_x, delta_y, delta_z, delta_yaw)
        if self.use_can_bus and self.use_fourier:
            self.fourier_embed_canbus = Fourier_Embed(len(self.can_bus_dims), self.fourier_nhidden)
        # vel_steering
        self.use_vel_steering = use_vel_steering
        self.vel_steering_dims = 4        # (vx, vy, v_yaw, steering)
        if self.use_vel_steering and self.use_fourier:
            self.fourier_embed_velsteering = Fourier_Embed(self.vel_steering_dims, self.fourier_nhidden)
        # vel
        self.use_vel = use_vel
        self.vel_dims = 2        # (vx, vy)
        if self.use_vel and self.use_fourier:
            self.fourier_embed_vel = Fourier_Embed(self.vel_dims, self.fourier_nhidden)
        # steering
        self.use_steering = use_steering
        self.steering_dims = 1        # (steering)
        if self.use_steering and self.use_fourier:
            self.fourier_embed_steering = Fourier_Embed(self.steering_dims, self.fourier_nhidden)
        # command
        self.use_command = use_command
        self.command_dims = 1             # command
        if self.use_command and self.use_fourier:
            self.fourier_embed_command = Fourier_Embed(self.command_dims, self.fourier_nhidden)
        # plan_traj
        self.use_plan_traj = use_plan_traj
        self.plan_traj_dims = 2
        if self.use_plan_traj and self.use_fourier:
            self.fourier_embed_plantraj = Fourier_Embed(self.plan_traj_dims, self.fourier_nhidden)
        # action_condition
        if self.use_fourier:
            self.action_condition_dims = 256 * use_can_bus + 256 * use_vel_steering + 256 * use_vel + 256 * use_steering + 256 * use_command + 256 * use_plan_traj
        elif not self.use_fourier:
            self.action_condition_dims = 4 * use_can_bus + 4 * use_vel_steering + 2 * use_vel + 1 * use_steering + 1 * use_command + 2 * use_plan_traj

        # Network configurations.
        self.num_pred_fcs = num_pred_fcs
        # How many bins predicted at the height dimensions.
        # By default, 1 for BEV prediction.
        self.num_pred_height = num_pred_height

        # build prev_render_neck
        if prev_render_neck is not None:
            self.prev_render_neck = builder.build_head(prev_render_neck)
        else:
            self.prev_render_neck = None

        # build transformer architecture.
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        # set evaluation configurations.
        self.eval_within_grid = eval_within_grid
        self._init_layers()
        
        if self.sem_norm: # occ pred_head
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs):
                sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.num_pred_height * self.num_classes))   # C -> cls
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)

            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 32
            self.mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_classes, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.ones_(self.mlp_gamma.bias)
        nn.init.ones_(self.mlp_beta.bias)

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # BEV query for the next frame.
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # Embeds for previous frame number.
        self.prev_frame_embedding = nn.Parameter(torch.Tensor(self.memory_queue_len, self.embed_dims))
        # Embeds for CanBus information.
        # Use position & orientation information of next frame's canbus.
        if self.use_can_bus or self.use_command or self.use_vel_steering or self.use_vel or self.use_steering or self.use_plan_traj:
            can_bus_input_dim = self.action_condition_dims
            self.fusion_mlp = nn.Sequential(
                nn.Linear(can_bus_input_dim, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.fusion_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.prev_frame_embedding)
            xavier_init(self.fusion_mlp, distribution='uniform', bias=0.)
        except:
            pass


    @auto_fp16(apply_to=('prev_features'))
    def _get_next_bev_features(self, prev_features, img_metas, target_frame_index, 
                               action_condition_dict, cond_norm_dict, tgt_points, ref_points, bev_h, bev_w):
        """ Forward function for each frame.

        Args:
            prev_features (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            target_frame_index: next frame information.
                For indexing target can_bus information.
            tgt_points (Tensor): query point coordinates in target frame coordinates.
            ref_points (Tensor): query point coordinates in previous frame coordinates.
        """
        # 1. BEV query
        bs, num_frames, _, emebd_dim = prev_features.shape
        dtype = prev_features.dtype
        #  * BEV queries.
        bev_queries = self.bev_embedding.weight.to(dtype)  # bev_h * bev_w, bev_dims
        bev_queries = bev_queries.unsqueeze(0)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w


        # 2. action condition
        action_condition = None
        plan_traj = action_condition_dict['plan_traj']
        command, vel_steering = action_condition_dict['command'][:, target_frame_index], action_condition_dict['vel_steering'][:, target_frame_index]
        if self.use_can_bus:
            #   * Can-bus information.
            cur_can_bus = [img_meta['future_can_bus'][target_frame_index] for img_meta in img_metas]
            cur_can_bus = np.array(cur_can_bus)[:, self.can_bus_dims]  # bs, 18
            cur_can_bus = torch.from_numpy(cur_can_bus).to(dtype).to(bev_pos.device)    # bs,4  (delta_x, delta_y, delta_z, delta_yaw)
            # fourier embed
            if self.use_fourier:
                cur_can_bus = self.fourier_embed_canbus(cur_can_bus)
            action_condition = cur_can_bus

        elif self.use_plan_traj:
            #  * Plan Traj
            cur_can_bus = plan_traj[:, -1, :2].float() # bs, 2
            # fourier embed
            if self.use_fourier:
                cur_can_bus = self.fourier_embed_plantraj(cur_can_bus)
            action_condition = cur_can_bus

        if self.use_command:
            command = command.unsqueeze(0)  # bs,1 (command)
            # fourier embed
            if self.use_fourier:
                command = self.fourier_embed_command(command)
            if action_condition is None:
                action_condition = command
            else:
                action_condition = torch.cat([action_condition, command], dim=-1)

        if self.use_vel_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            # fourier embed
            if self.use_fourier:
                vel_steering = self.fourier_embed_velsteering(vel_steering)
            if action_condition is None:
                action_condition = vel_steering
            else:
                action_condition = torch.cat([action_condition, vel_steering], dim=-1)

        if self.use_vel:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            vel = vel_steering[:, :self.vel_dims]
            # fourier embed
            if self.use_fourier:
                vel = self.fourier_embed_vel(vel)
            if action_condition is None:
                action_condition = vel
            else:
                action_condition = torch.cat([action_condition, vel], dim=-1)

        if self.use_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            steering = vel_steering[:, -1:]
            # fourier embed
            if self.use_fourier:
                steering = self.fourier_embed_steering(steering)
            if action_condition is None:
                action_condition = steering
            else:
                action_condition = torch.cat([action_condition, steering], dim=-1)

        if action_condition is not None:
            action_condition = self.fusion_mlp(action_condition.float())

        #  * sum different query embedding together.
        #    (bs, bev_h * bev_w, dims)
        if (self.use_can_bus or self.use_plan_traj) and self.condition_ca_add == 'add':
            bev_queries_input = bev_queries + action_condition.unsqueeze(1)
        else:
            bev_queries_input = bev_queries


        # 3. obtain prev embeddings (bs, num_frames, bev_h * bev_w, dims).
        if self.prev_render_neck:
            from .visualization import save_bev_render_compare
            import time
            render_dict = self.prev_render_neck(prev_features.view(bs, num_frames, bev_w, bev_h, emebd_dim).transpose(2,3), cond_norm_dict)
            save_bev_render_compare(prev_features, render_dict['bev_embed'], bs, num_frames, bev_w, bev_h, emebd_dim, prefix=f'bev_render_{time.strftime("%Y%m%d_%H%M%S")}')
            prev_features = render_dict['bev_embed']
            bev_sem_pred = render_dict['bev_sem_pred']  # B,cls,H,W

        frame_embedding = self.prev_frame_embedding
        prev_features_input = (prev_features +
                               frame_embedding[None, :, None, :])

        # 4. do transformer layers to get BEV features.
        next_bev_feat = self.transformer(
            prev_features_input,
            bev_queries_input,
            tgt_points=tgt_points,
            ref_points=ref_points,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
            action_condition=action_condition,
        )


        return next_bev_feat, bev_sem_pred

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self,
                prev_feats,
                img_metas,
                target_frame_index,
                action_condition_dict,
                cond_norm_dict,
                tgt_points,  # tgt_points config for self-attention.
                ref_points,  # ref_points config for cross-attention.
                bev_h, bev_w,):
        f"""Forward function: a wrapper function for self._get_next_bev_features
        
        From previous multi-frame BEV features (mlvl_feats) predict 
        the next-frame of point cloud.
        Args:
            mlvl_feats (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            tgt_points (Tensor): query point coordinates in target frame coordinates.
            ref_points (Tensor): query point coordinates in previous frame coordinates.
        Returns:
            A dict with:
                next_bev_feature (Tensor): prediction of BEV features of next frame.
                next_bev_pred (Tensor): prediction of next BEV occupancy (Freespace).
        """
        bs, num_frames, bev_grids_num, bev_dims = prev_feats.shape
        assert bev_dims == self.embed_dims
        assert bev_h * bev_w == bev_grids_num
        assert bev_h * bev_w == tgt_points.shape[1]

        next_bev_feat, bev_sem_pred = self._get_next_bev_features(
            prev_feats, img_metas, target_frame_index, action_condition_dict, 
            cond_norm_dict, tgt_points, ref_points, bev_h, bev_w)
        return next_bev_feat, bev_sem_pred

    def forward_head(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num, inter_num, bs, bev_h * bev_w, dims]
        """
        pass


@HEADS.register_module()
class WorldHeadBase(WorldHeadTemplate):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)