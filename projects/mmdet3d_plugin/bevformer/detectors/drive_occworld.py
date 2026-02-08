import mmcv
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
import os
import torch.nn.functional as F
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.losses.plan_reg_loss_lidar import plan_reg_loss
from projects.mmdet3d_plugin.bevformer.utils.metric_stp3 import PlanningMetric
from projects.mmdet3d_plugin.bevformer.utils.planning_metrics import PlanningMetric_v2
from torchvision.transforms.functional import rotate

from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils
from ..utils.timer import avg_sections, CallSectionsTimer 
from contextlib import contextmanager


@DETECTORS.register_module()
class Drive_OccWorld(BEVFormer):
    def __init__(self,
                 # Future predictions.
                 future_pred_head,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Plan Head
                 turn_on_plan=False,
                 plan_head=None,
                 supervision_type='none',

                 # Memory Queue configurations.
                 memory_queue_len=1,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 only_generate_dataset=False,
                 supervise_all_future=True,
                 use_lwm=False,

                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)

        # Define self.t in __init__, no globals involved.
        self._current_timer: Optional[CallSectionsTimer] = None
        @contextmanager
        def _noop():
            # No-op context when no timer is active
            yield
        def section(name: str):
            # If a timer is active, delegate; otherwise, no-op
            return self._current_timer(name) if self._current_timer else _noop()
        self.t = section  # now you can: with self.t("stage"):

        # occ head
        self.future_pred_head = builder.build_head(future_pred_head)

        # plan head
        self.turn_on_plan = turn_on_plan
        if turn_on_plan:
            self.plan_head = builder.build_head(plan_head)
            self.plan_head_type = plan_head.type
            self.planning_metric = None
            self.n_future = 6
            self.planning_metric_v2 = PlanningMetric_v2(n_future=self.n_future)
        
        # memory queue
        self.memory_queue_len = memory_queue_len

        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.only_generate_dataset = only_generate_dataset
        self.supervise_all_future = supervise_all_future
        self.use_lwm = use_lwm

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path

        # remove the useless modules in pts_bbox_head
        #  * box/cls prediction head; decoder transformer.
        del self.pts_bbox_head.cls_branches, self.pts_bbox_head.reg_branches
        del self.pts_bbox_head.query_embedding
        del self.pts_bbox_head.transformer.decoder

        if self.only_train_cur_frame:
            # remove useless parameters.
            del self.future_pred_head.transformer
            del self.future_pred_head.bev_embedding
            del self.future_pred_head.prev_frame_embedding
            del self.future_pred_head.can_bus_mlp
            del self.future_pred_head.positional_encoding

        self.supervision_type = supervision_type
        if self.turn_on_plan and supervision_type == 'only_plan':
            for name, param in self.named_parameters():
                if 'plan_head' not in name:
                    param.requires_grad = False


    def set_epoch(self, epoch):
        self.training_epoch = epoch

    ####################### Image Feature Extraction. #######################
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if ('aug_param' in img_metas[0] and
                img_metas[0]['aug_param'] is not None and
                img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True):
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]

        return img_feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask and self.grid_mask_image:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if self.use_grid_mask and self.grid_mask_backbone_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            if self.use_grid_mask and self.grid_mask_fpn_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, prev_img_metas, ref_img_metas):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_num_frames = 1
        history_num_frames = num_frames - ref_num_frames
        # history
        ref_to_history_list = []
        for img_metas in prev_img_metas:
            img_metas_len = len(img_metas)
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(img_metas_len-history_num_frames, img_metas_len)]   # [-3,-2,-1]
            ref_to_history_list.append(cur_ref_to_prev)
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))
        # ref
        ref_to_ref_list = []
        for img_metas in ref_img_metas:
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(ref_num_frames)]
            ref_to_ref_list.append(cur_ref_to_prev)
        ref_to_ref_list = tensor.new_tensor(np.array(ref_to_ref_list))
        # concat
        if ref_to_history_list.shape[1] == 0:   # not use history
            ref_to_history_list = ref_to_ref_list
        else:
            ref_to_history_list = torch.cat([ref_to_history_list, ref_to_ref_list], dim=1)
        return ref_to_history_list

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas, plan_traj):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation metric from reference to each history frames.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
        """
        bs, num_frame = ref_to_history_list.shape[:2]
        translation_xy = torch.cumsum(plan_traj, dim=1)[:, -1, :2].float()

        # 1. get future2ref and ref2future_matrix of frame_idx.
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))
        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            future2ref = future2ref.transpose(-1, -2)
            future2ref[:, :2, 3] = translation_xy
            future2ref = future2ref.transpose(-1, -2)
            future2ref = future2ref.detach().clone()

        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))
        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            ref2future = ref2future.transpose(-1, -2)
            rot = ref2future[:, :3, :3]
            translation_xyz = future2ref[:, 3, :3].unsqueeze(2)     # cur2ref
            translation_xyz = -(rot @ translation_xyz).squeeze(2)   # ref2cur
            ref2future[:, :3, 3] = translation_xyz
            ref2future = ref2future.transpose(-1, -2)
            ref2future = ref2future.detach().clone()

        # 2. compute the transformation matrix from current frame to all previous frames.
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        bev_grids = e2e_predictor_utils.get_bev_grids(
            self.bev_h, self.bev_w, bs * num_frame)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)
        bev_coords = e2e_predictor_utils.bev_grids_to_coordinates(
            bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        aligned_bev_coords = torch.cat([
            bev_coords, torch.ones_like(bev_coords[..., :2])], -1)
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]
        aligned_bev_grids, _ = e2e_predictor_utils.bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # range of [0, 1]
        # b, h*w, num_frame, 2
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        tgt_grids = bev_grids[:, -1].contiguous()
        return tgt_grids, aligned_bev_grids, ref2future, future_to_history_list.transpose(-1, -2)
    

    def obtain_ref_bev(self, img, img_metas, prev_bev):
        # Extract current BEV features.
        # C1. Forward.
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # C3. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
        return ref_bev

    
    def future_pred(self, prev_bev_input, action_condition_dict, cond_norm_dict, plan_dict, 
                    valid_frames, img_metas, prev_img_metas, num_frames, future_bev=None):
        
        # D1. preparations.
        # prev_bev_input: B,memory_queue_len,HW,C
        ref_bev = prev_bev_input[:, -1].unsqueeze(0).repeat(
                len(self.future_pred_head.bev_pred_head), 1, 1, 1).contiguous()

        next_bev_feats, next_bev_sem, next_pose_loss = [ref_bev], [], []
        next_pose_preds = plan_dict['ref_pose_pred'] # B,Lout,2


        # D2. Align previous frames to the reference coordinates.
        ref_img_metas = [[each[num_frames-1]] for each in prev_img_metas]
        prev_img_metas = [[each[i] for i in range(num_frames-1)] for each in prev_img_metas]
        ref_to_history_list = self._get_history_ref_to_previous_transform(
            prev_bev_input, prev_bev_input.shape[1], prev_img_metas, ref_img_metas)


        # D3. future decoder forward.
        if self.training:
            future_frame_num = self.future_pred_frame_num
        else:
            future_frame_num = self.test_future_frame_num

        for future_frame_index in range(1, future_frame_num + 1):
            if not self.turn_on_plan or (self.turn_on_plan and self.training and self.training_epoch < 12):
                plan_traj = plan_dict['gt_traj'][:, :future_frame_index, :2]
            else:
                plan_traj = next_pose_preds

            if self.supervision_type == 'only_plan':
                plan_traj = next_pose_preds
                
            action_condition_dict['plan_traj'] = plan_traj

            # 1. obtain the coordinates of future BEV query to previous frames.
            tgt_grids, aligned_prev_grids, ref2future, future2history = self._align_bev_coordnates(
                future_frame_index, ref_to_history_list, img_metas, plan_traj)
            cond_norm_dict['future2history'] = future2history


            # 2. transform for generating freespace of future frame.
            # pred_feat: inter_num, bs, bev_h * bev_w, c
            if future_frame_index in valid_frames:  # compute loss if it is a valid frame.
                pred_feat, bev_sem_pred = self.future_pred_head(
                    prev_bev_input, img_metas, future_frame_index, action_condition_dict, cond_norm_dict,
                    tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
                next_bev_feats.append(next_bev_feats[-1] + pred_feat) # TODO: predict residual bev features
                next_bev_sem.append(bev_sem_pred)
            else:
                with torch.no_grad():
                    pred_feat, bev_sem_pred = self.future_pred_head(
                        prev_bev_input, img_metas, future_frame_index, action_condition_dict, cond_norm_dict,
                        tgt_points=tgt_grids, bev_h=self.bev_h, bev_w=self.bev_w, ref_points=aligned_prev_grids)
                    next_bev_feats.append(next_bev_feats[-1] + pred_feat) # TODO: predict residual bev features

            if self.turn_on_plan and next_pose_preds.shape[1] < self.n_future:
                # sample_traj  gt_traj
                sample_traj_i,  gt_traj_i = plan_dict['sample_traj'][:,:,future_frame_index], plan_dict['gt_traj'][:,future_frame_index]
                # command
                command_i = action_condition_dict['command'][:,future_frame_index]
                # forward plan_head 
                # sem_occupancy
                if 'v1' in self.plan_head_type:
                    if plan_dict['sem_occupancy'] is None:
                        sem_occupancy_i = self.future_pred_head.forward_head(next_bev_feats[-1].unsqueeze(0))[-1, -1].argmax(-1).detach()
                        bs, hw, d = sem_occupancy_i.shape
                        sem_occupancy_i = sem_occupancy_i.view(bs, self.bev_w, self.bev_h, d).transpose(1,2)
                    else:
                        sem_occupancy_i = plan_dict['sem_occupancy'][:,future_frame_index]

                    pose_pred, pose_loss = self.plan_head(next_bev_feats[-1][-1], sample_traj_i, sem_occupancy_i, command_i, gt_traj_i)
                    # update prev_pose and store pred
                    next_pose_loss.append(pose_loss)
                elif 'v2' in self.plan_head_type:
                    pose_pred = self.plan_head(next_bev_feats[-1][-1], command_i)

                next_pose_preds = torch.cat([next_pose_preds, pose_pred], dim=1)

            # 4. update pred_feat to prev_bev_input and update ref_to_history_list.
            prev_bev_input = torch.cat([prev_bev_input, next_bev_feats[-1][-1].unsqueeze(1)], 1) # TODO: predict residual bev features
            prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
            # update ref2future to ref_to_history_list.
            ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
            ref_to_history_list = ref_to_history_list[:, 1:].contiguous()
            # update occ_gts
            if cond_norm_dict['occ_gts'] is not None:
                cond_norm_dict['occ_gts'] = cond_norm_dict['occ_gts'][:, 1:, ...].contiguous()


        # D4. forward head.
        next_bev_feats = torch.stack(next_bev_feats, 0)
        # # forward head
        # next_bev_preds = self.future_pred_head.forward_head(next_bev_feats)

        ret_dict = {}
        # ret_dict = {
        #     'next_bev_preds': next_bev_preds,
        #     'next_bev_sem': next_bev_sem
        # }
        if future_bev is not None and self.use_lwm:
            loss_bev = self.compute_bev_loss(next_bev_feats[1:], torch.stack(future_bev).detach())
            ret_dict['loss_bev'] = loss_bev 
        
        if self.turn_on_plan:
            ret_dict['next_pose_preds'] = next_pose_preds
            ret_dict['next_pose_loss'] = next_pose_loss
        return ret_dict

    def compute_bev_loss(self, next_bev_feats, future_bev):
        losses_bev = {}
        factor = 1 / next_bev_feats.shape[1]
        for i in range(next_bev_feats.shape[1]):
            losses_bev[f'loss_bev_lwm_{i}'] = factor * torch.nn.functional.mse_loss(next_bev_feats[:, i], future_bev)
        return losses_bev

    def compute_occ_loss(self, occ_preds, occ_gts):
        # preds [Lout, inter_num, bs, bev_h * bev_w, d, num_cls]    Lout = cur + future_select
        occ_preds = occ_preds.permute(1, 0, 2, 5, 3, 4)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4) # TODO: now the feature map size is the same as the output size, not efficient!
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])
        
        # occ loss
        losses_occupancy = self.future_pred_head.loss_occ(occ_preds, occ_gts)
        return losses_occupancy
    
    def compute_sem_norm_loss(self, bev_sem_preds, occ_gts):
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length+1-self.memory_queue_len:-1]

        # loss sem
        if bev_sem_preds[0] is not None:
            bev_sem_preds = torch.stack(bev_sem_preds, dim=0).transpose(0,1)
            loss_sem_norm = self.future_pred_head.loss_sem_norm(bev_sem_preds, occ_gts)
        return loss_sem_norm

    def compute_plan_loss(self, outs_planning, sdc_planning, sdc_planning_mask, gt_future_boxes):
        ## outs_planning, sdc_planning: under ref_lidar coord
        pred_under_ref = torch.cumsum(outs_planning, dim=1)
        gt_under_ref = torch.cumsum(sdc_planning, dim=1)

        losses_plan = self.plan_head.loss(pred_under_ref, gt_under_ref, sdc_planning_mask, gt_future_boxes)
        return losses_plan
    
    def evaluate_occ(self, occ_preds, occ_gts, img_metas):
        # preds
        occ_preds = occ_preds.permute(1, 0, 2, 5, 3, 4)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])

        hist_for_iou = self.evaluate_occupancy_forecasting(occ_preds[-1], occ_gts, img_metas=img_metas, save_pred=self._viz_pcd_flag, save_path=self._viz_pcd_path)
        hist_for_iou_current = self.evaluate_occupancy_forecasting(occ_preds[-1][0:1], occ_gts[0:1], img_metas=img_metas, save_pred=False)
        hist_for_iou_future = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, save_pred=False)
        hist_for_iout_future_time_weighting = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, time_weighting=True)
        return hist_for_iou, hist_for_iou_current, hist_for_iou_future, hist_for_iout_future_time_weighting

    def evaluate_plan(self, next_pose_preds, sdc_planning, sdc_planning_mask, segmentation_bev, img_metas):
        """
            pred_ego_fut_trajs: B,Lout,2
            gt_ego_fut_trajs:   B,Lout,2
            segmentation_bev:   B,Lout,h,w
        """
        next_pose_gts = sdc_planning

        # pred, gt: under ref_lidar coord
        pred_under_ref = torch.cumsum(next_pose_preds[..., :2], dim=1)
        gt_under_ref = torch.cumsum(next_pose_gts[..., :2], dim=1).float()

        if self._viz_pcd_flag:
            save_data = np.load(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]+'.npz'), allow_pickle=True)
            np.savez(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]), 
                                occ_pred=save_data['occ_pred'], pose_pred=pred_under_ref[0].detach().cpu().numpy())

        self.planning_metric_v2(pred_under_ref, gt_under_ref, sdc_planning_mask, segmentation_bev)

    @auto_fp16(apply_to=('img', 'segmentation', 'flow', 'sdc_planning'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      # occ_flow
                      segmentation=None,
                      instance=None, 
                      flow=None,
                      # sdc-plan
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      # sample_traj
                      sample_traj=None,
                      # vel_sterring
                      vel_steering=None,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            segmentation (list[torch.Tensor])
            flow (list[torch.Tensor])
            sample_traj
        Returns:
            dict: Losses of different branches.
        """

        # manually stop forward
        if self.only_generate_dataset:
            return {"pseudo_loss": torch.tensor(0.0, device=img.device, requires_grad=True)}


        # Augmentations.
        # A1. Randomly drop cur image input.
        if np.random.rand() < self.random_drop_image_rate:
            img[:, -1:, ...] = torch.zeros_like(img[:, -1:, ...])
        # A2. Randomly drop previous image inputs.
        num_frames = img.size(1)
        if np.random.rand() < self.random_drop_prev_rate:
            random_drop_prev_v2_end_idx = (
                self.random_drop_prev_end_idx if self.random_drop_prev_end_idx is not None
                else num_frames)
            drop_prev_index = np.random.randint(
                self.random_drop_prev_start_idx, random_drop_prev_v2_end_idx)
        else:
            drop_prev_index = -1


        # Extract history BEV features.
        # B1. Forward previous frames.
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        # B2. Randomly grid-mask prev_bev.
        prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas, drop_prev_index=drop_prev_index)
        # B2. Randomly grid-mask prev_bev.
        if self.grid_mask_prev and prev_bev is not None:
            b, n, c = prev_bev.shape
            assert n == self.bev_h * self.bev_w
            prev_bev = prev_bev.view(b, self.bev_h, self.bev_w, c)
            prev_bev = prev_bev.permute(0, 3, 1, 2).contiguous()
            prev_bev = self.grid_mask(prev_bev)
            prev_bev = prev_bev.view(b, c, n).permute(0, 2, 1).contiguous()


        # C. Extract current BEV features.
        img = img[:, -1, ...]
        img_metas = [each[num_frames-1] for each in img_metas]
        ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev)

        sem_occupancy = None
        if self.turn_on_plan:
            ref_sample_traj, ref_real_traj, ref_command = sample_traj[:, :, 0], sdc_planning[:, 0], command[:, 0]
            # use pred_occupancy to calculate sample_traj cost during inference, GT_occupancy during training
            if 'v1' in self.plan_head_type:
                sem_occupancy = segmentation[0][self.future_pred_head.history_queue_length:].unsqueeze(0)   # using GT occupancy to calculate sample_traj cost during training
                sem_occupancy = F.interpolate(sem_occupancy, size=(self.bev_h, self.bev_w, self.future_pred_head.num_pred_height), mode='nearest')  # B,Lout,H,W,D
                ref_sem_occupancy = sem_occupancy[:, 0] # B,H,W,D
                ref_pose_pred, ref_pose_loss = self.plan_head(ref_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
            elif 'v2' in self.plan_head_type:
                ref_pose_pred = self.plan_head(ref_bev, ref_command)
        else:
            ref_pose_pred = None

        # TODO: get future bev features.
        future_bev = None
        if self.use_lwm:
            selected_bev = ref_bev
            future_bev = []
            for ind in range(1, kwargs['future_img'].shape[1]):
                with torch.no_grad():
                    selected_img = kwargs['future_img'][:, ind, ...]
                    selected_img_metas = [each[ind] for each in kwargs['future_img_metas']]
                    selected_bev = self.obtain_ref_bev(selected_img, selected_img_metas, selected_bev)
                    future_bev.append(selected_bev)

        # D. Extract future BEV features.
        valid_frames = [0]
        if not self.only_train_cur_frame:
            if self.supervise_all_future:
                valid_frames.extend(list(range(1, self.future_pred_frame_num + 1)))
            else:  # randomly select one future frame for computing loss to save memory cost.
                train_frame = np.random.choice(np.arange(1, self.future_pred_frame_num + 1), 1)[0]
                valid_frames.append(train_frame)
            # D1. prepare memory_queue
            prev_bev_list = torch.stack(prev_bev_list, dim=1)
            prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...]
            # D2. prepare conditional-normalization dict
            if self.future_pred_head.prev_render_neck and self.future_pred_head.prev_render_neck.sem_norm and self.future_pred_head.prev_render_neck.sem_gt_train and self.training_epoch < 12:
                occ_gts = segmentation[0][self.future_pred_head.history_queue_length+1-self.memory_queue_len:-1]
                occ_gts = F.interpolate(occ_gts.unsqueeze(1), size=(self.bev_h, self.bev_w, self.future_pred_head.prev_render_neck.pred_height), mode='nearest').transpose(0,1)
            else:
                occ_gts = None
            cond_norm_dict = {'occ_gts': occ_gts}
            # D3. prepare action condition dict
            action_condition_dict = {'command':command, 'vel_steering': vel_steering}
            # D4. prepare planning dict
            plan_dict = {'sem_occupancy': sem_occupancy, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred}

            # D5. predict future occ in auto-regressive manner
            ret_dict = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict, 
                                                                            valid_frames, img_metas, prev_img_metas, num_frames, future_bev=future_bev)
        # E. Compute Loss
        losses = dict()
        # E1. Compute loss for occ predictions.
        losses_occupancy = self.compute_occ_loss(ret_dict['next_bev_preds'], segmentation)
        losses.update(losses_occupancy)
        if self.use_lwm:
            losses.update(ret_dict['loss_bev'])

        # E3. Compute loss for plan regression.
        if self.turn_on_plan:
            gt_future_boxes = gt_future_boxes[0]   # Lout,[boxes]  NOTE: Current Support bs=1
            losses_plan = self.compute_plan_loss(ret_dict['next_pose_preds'], sdc_planning[:, :self.n_future], sdc_planning_mask[:, :self.n_future], gt_future_boxes[:self.n_future])
            if 'v1' in self.plan_head_type:
                losses_plan_cost = ref_pose_loss + sum(ret_dict['next_pose_loss'])
                losses_plan.update(losses_plan_cost = 0.1 * losses_plan_cost)
            losses.update(losses_plan)

        # E4. Compute loss for bev rendering
        if self.future_pred_head.prev_render_neck.sem_norm:
            losses_bev_render = self.compute_sem_norm_loss(ret_dict['next_bev_sem'], segmentation)
            losses.update(losses_bev_render)

        return losses

    @avg_sections(name="test", unit="ms", warmup=2, sync=torch.cuda.synchronize)
    def forward_test(self, 
                     img_metas, 
                     img=None,
                     # occ_flow
                     segmentation=None, 
                     instance=None, 
                     flow=None, 
                     # sdc-plan
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     segmentation_bev=None,
                     # sample_traj
                     sample_traj=None,
                     # vel_sterring
                     vel_steering=None,
                     **kwargs):
        """has similar implementation with train forward."""

        # manually stop forward
        if self.only_generate_dataset:
            return {'hist_for_iou': 0, 'pred_c': 0, 'vpq':0}


        self.eval()
        # Extract history BEV features.
        # B. Forward previous frames.
        num_frames = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        with self.t("history"):
            prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas)
        
        with self.t("current"):

            # C. Extract current BEV features.
            img = img[:, -1, ...]
            img_metas = [each[num_frames-1] for each in img_metas]
            ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev)

        with self.t('planning'):
            if self.turn_on_plan:
                ref_sample_traj, ref_real_traj, ref_command = sample_traj[:, :, 0], sdc_planning[:, 0], command[:, 0]
                # use pred_occupancy to calculate sample_traj cost during inference, GT_occupancy during training
                if 'v1' in self.plan_head_type:
                    sem_occupancy = segmentation[0][self.future_pred_head.history_queue_length:].unsqueeze(0)   # using GT occupancy to calculate sample_traj cost during training
                    sem_occupancy = F.interpolate(sem_occupancy, size=(self.bev_h, self.bev_w, self.future_pred_head.num_pred_height), mode='nearest')  # B,Lout,H,W,D
                    ref_sem_occupancy = sem_occupancy[:, 0] # B,H,W,D
                    ref_pose_pred, ref_pose_loss = self.plan_head(ref_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
                elif 'v2' in self.plan_head_type:
                    ref_pose_pred = self.plan_head(ref_bev, ref_command)
            else:
                ref_pose_pred = None

            # D. Predict future BEV.
            valid_frames = [] # no frame have grad
            # D1. prepare memory_queue
            prev_bev_list = torch.stack(prev_bev_list, dim=1)
            prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...]
            # D2. prepare conditional-normalization dict
            cond_norm_dict = {'occ_gts': None}
            # D3. prepare action condition dict
            action_condition_dict = {'command':command, 'vel_steering': vel_steering}
            # D4. prepare planning dict
            plan_dict = {'sem_occupancy': None, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred}

            # D5. predict future occ in auto-regressive manner
            ret_dict = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                                    valid_frames, img_metas, prev_img_metas, num_frames)

        return {}
        # E. Evaluate
        test_output = {}
        # evaluate occ
        occ_iou, occ_iou_current, occ_iou_future, occ_iou_future_time_weighting = self.evaluate_occ(ret_dict['next_bev_preds'], segmentation, img_metas)
        test_output.update(hist_for_iou=occ_iou, hist_for_iou_current=occ_iou_current, 
                           hist_for_iou_future=occ_iou_future, hist_for_iou_future_time_weighting=occ_iou_future_time_weighting)

        # evluate plan
        if self.turn_on_plan:
            self.evaluate_plan(ret_dict['next_pose_preds'], sdc_planning[:, :self.n_future], sdc_planning_mask[:, :self.n_future], segmentation_bev[:, :self.n_future], img_metas)

        return test_output

    def evaluate_occupancy_forecasting(self, pred, gt, img_metas=None, save_pred=False, save_path=None, time_weighting=False):

        B, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()

        hist_all = 0
        iou_per_pred_list = []
        pred_list = []
        gt_list = []
        for i in range(B):
            pred_cur = pred[i,...]
            pred_cur = torch.argmax(pred_cur, dim=0).cpu().numpy()
            gt_cur = gt[i, ...].cpu().numpy()
            gt_cur = gt_cur.astype(np.int)

            pred_list.append(pred_cur)
            gt_list.append(gt_cur)

            # ignore noise
            noise_mask = gt_cur != 255

            # GMO and others for max_label=2
            # multiple movable objects for max_label=9
            hist_cur, iou_per_pred = fast_hist(pred_cur[noise_mask], gt_cur[noise_mask], max_label=self.future_pred_head.num_classes)
            if time_weighting:
                hist_all = hist_all + 1 / (i+1) * hist_cur
            else:
                hist_all = hist_all + hist_cur
            iou_per_pred_list.append(iou_per_pred)

        # whether save prediction results
        if save_pred:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            pred_for_save_list = []
            for k in range(B):
                pred_for_save = torch.argmax(pred[k], dim=0).cpu()
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1, 1).expand(H, W, D)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W, 1).expand(H, W, D)
                z_grid = torch.linspace(0, D-1, D, dtype=torch.long)
                z_grid = z_grid.view(1, 1, D).expand(H, W, D)
                segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 3)
                segmentation_label = pred_for_save.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1) # N,4
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept].cpu().numpy()
                pred_for_save_list.append(segmentation_for_save)
            np.savez(os.path.join(save_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]), occ_pred=pred_for_save_list)

        return hist_all

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2) 
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred
