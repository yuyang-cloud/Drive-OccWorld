import copy
import torch
import numpy as np

from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.parallel import DataContainer as DC

from .nuscenes_world_dataset_template import NuScenesWorldDatasetTemplate


@DATASETS.register_module()
class NuScenesWorldDatasetV1(NuScenesWorldDatasetTemplate):
    def _mask_points(self, pts_list):
        assert self.ego_mask is not None
        # remove points belonging to ego vehicle.
        masked_pts_list = []
        for pts in pts_list:
            ego_mask = np.logical_and(
                np.logical_and(self.ego_mask[0] <= pts[:, 0],
                               self.ego_mask[2] >= pts[:, 0]),
                np.logical_and(self.ego_mask[1] <= pts[:, 1],
                               self.ego_mask[3] >= pts[:, 1]),
            )
            pts = pts[np.logical_not(ego_mask)]
            masked_pts_list.append(pts)
        pts_list = masked_pts_list
        return pts_list

    def union2one(self, previous_queue, future_queue):
        """
            previous_queue = history_len*['img', 'points', 'aug_param']  include current frame
            future_queue = future_len*['img', 'points', 'aug_param']     include current frame
        """
        # 1. get transformation from all frames to current (reference) frame
        ref_meta = previous_queue[-1]['img_metas'].data
        valid_scene_token = ref_meta['scene_token']
        # compute reference e2g_transform and g2e_transform.
        ref_e2g_translation = ref_meta['ego2global_translation']
        ref_e2g_rotation = ref_meta['ego2global_rotation']
        ref_e2g_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=False)
        ref_g2e_transform = transform_matrix(
            ref_e2g_translation, Quaternion(ref_e2g_rotation), inverse=True)
        # compute reference l2e_transform and e2l_transform
        ref_l2e_translation = ref_meta['lidar2ego_translation']
        ref_l2e_rotation = ref_meta['lidar2ego_rotation']
        ref_l2e_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=False)
        ref_e2l_transform = transform_matrix(
            ref_l2e_translation, Quaternion(ref_l2e_rotation), inverse=True)

        queue = previous_queue[:] + future_queue[1:]  # total_len: 4(history)+1(current)+2(future)
        total_cur2ref_lidar_transform = []  # total_len*[4*4]
        total_ref2cur_lidar_transform = []  # total_len*[4*4]
        for i, each in enumerate(queue):
            meta = each['img_metas'].data

            # store the transformation from current frame to reference frame.
            curr_e2g_translation = meta['ego2global_translation']
            curr_e2g_rotation = meta['ego2global_rotation']
            curr_e2g_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=False)
            curr_g2e_transform = transform_matrix(
                curr_e2g_translation, Quaternion(curr_e2g_rotation), inverse=True)

            curr_l2e_translation = meta['lidar2ego_translation']
            curr_l2e_rotation = meta['lidar2ego_rotation']
            curr_l2e_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=False)
            curr_e2l_transform = transform_matrix(
                curr_l2e_translation, Quaternion(curr_l2e_rotation), inverse=True)

            # compute future to reference matrix.
            cur_lidar_to_ref_lidar = (curr_l2e_transform.T @
                                      curr_e2g_transform.T @
                                      ref_g2e_transform.T @
                                      ref_e2l_transform.T)
            total_cur2ref_lidar_transform.append(cur_lidar_to_ref_lidar)

            # compute reference to future matrix.
            ref_lidar_to_cur_lidar = (ref_l2e_transform.T @
                                      ref_e2g_transform.T @
                                      curr_g2e_transform.T @
                                      curr_e2l_transform.T)
            total_ref2cur_lidar_transform.append(ref_lidar_to_cur_lidar)


        # 2. Parse previous and future can_bus information.
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        ref_meta = previous_queue[-1]['img_metas'].data

        # 2.1 Previous
        for i, each in enumerate(previous_queue):
            metas_map[i] = each['img_metas'].data

            if 'aug_param' in each:
                metas_map[i]['aug_param'] = each['aug_param']
            
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Set the original point of this motion.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                metas_map[i]['can_bus'] = new_can_bus
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # Compute the later waypoint.
                # To align the shift and rotate difference due to the BEV.
                new_can_bus = copy.deepcopy(metas_map[i]['can_bus'])
                new_can_bus[:3] = tmp_pos - prev_pos
                new_can_bus[-1] = tmp_angle - prev_angle
                metas_map[i]['can_bus'] = new_can_bus
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

            # compute cur_lidar_to_ref_lidar transformation matrix for quickly align generated
            #  bev features to the reference frame.
            metas_map[i]['ref_lidar_to_cur_lidar'] = total_ref2cur_lidar_transform[i]

        # 2.2. Future
        future_metas_map = {}
        current_scene_token = ref_meta['scene_token']
        ref_can_bus = None
        future_can_bus = []
        future2ref_lidar_transform = []
        ref2future_lidar_transform = []
        for i, each in enumerate(future_queue):
            future_meta = each['img_metas'].data
            if future_meta['scene_token'] != current_scene_token:
                break

            # store the transformation:
            future2ref_lidar_transform.append(
                total_cur2ref_lidar_transform[i + len(previous_queue) - 1]
            )  # current -> reference.
            ref2future_lidar_transform.append(
                total_ref2cur_lidar_transform[i + len(previous_queue) - 1]
            )  # reference -> current.

            # can_bus information.
            if i == 0:
                prev_pos = copy.deepcopy(future_meta['can_bus'][:3])
                prev_angle = copy.deepcopy(future_meta['can_bus'][-1])
                new_can_bus = copy.deepcopy(future_meta['can_bus'])
                new_can_bus[:3] = 0
                new_can_bus[-1] = 0
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])
            else:
                future_metas_map[i] = copy.deepcopy(future_meta)
                if 'aug_param' in each:
                    future_metas_map[i]['aug_param'] = each['aug_param']
                future_metas_map[i]['prev_bev_exists'] = True

                # To align the shift and rotate difference due to the BEV.
                tmp_pos = copy.deepcopy(future_metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(future_metas_map[i]['can_bus'][-1])
                new_can_bus = copy.deepcopy(future_metas_map[i]['can_bus'])
                new_can_bus[:3] = tmp_pos - prev_pos
                new_can_bus[-1] = tmp_angle - prev_angle
                future_metas_map[i]['can_bus'] = new_can_bus
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

                new_can_bus = copy.deepcopy(future_meta['can_bus'])

                new_can_bus_pos = np.array([0, 0, 0, 1]).reshape(1, 4)
                ref2prev_lidar_transform = ref2future_lidar_transform[-2]
                cur2ref_lidar_transform = future2ref_lidar_transform[-1]
                new_can_bus_pos = new_can_bus_pos @ cur2ref_lidar_transform @ ref2prev_lidar_transform

                new_can_bus_angle = new_can_bus[-1] - ref_can_bus[-1]
                new_can_bus[:3] = new_can_bus_pos[:, :3]
                new_can_bus[-1] = new_can_bus_angle
                future_can_bus.append(new_can_bus)
                ref_can_bus = copy.deepcopy(future_meta['can_bus'])

        # save to metas_map
        metas_map[len(previous_queue) - 1]['future_can_bus'] = np.array(future_can_bus)
        metas_map[len(previous_queue) - 1]['future2ref_lidar_transform'] = (
            np.array(future2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['ref2future_lidar_transform'] = (
            np.array(ref2future_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_cur2ref_lidar_transform'] = (
            np.array(total_cur2ref_lidar_transform))
        metas_map[len(previous_queue) - 1]['total_ref2cur_lidar_transform'] = (
            np.array(total_ref2cur_lidar_transform))
        ret_queue = previous_queue[-1]
        ret_queue['img_metas'] = DC(metas_map, cpu_only=True)
        ret_queue.pop('aug_param', None)


        # 3. Prepare image inputs of history and current frames.
        imgs_list = [each['img'].data for each in previous_queue]   # history_len*[imgs]
        ret_queue['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)

        # TODO: add future images and metas
        future_imgs_list = [each['img'].data for each in future_queue]   # future_len*[imgs]
        ret_queue['future_img'] = DC(torch.stack(future_imgs_list), cpu_only=False, stack=True)
        ret_queue['future_img_metas'] = DC(future_metas_map, cpu_only=True)

        # 4. Prepare GT occupancy and flow of current and future frames (saved in the current frame).
        if self.use_fine_occ:
            segmentation_list = queue[self.queue_length]['gt_occ'].data
            ret_queue.pop('gt_occ', None)
        else:
            segmentation_list = queue[self.queue_length]['segmentation'].data
            instance_list = queue[self.queue_length]['instance'].data if 'instance' in queue[self.queue_length].keys() else None
            flow_list = queue[self.queue_length]['flow'].data if self.turn_on_flow and not self.test_mode else None
        # save to ret_queue
        ret_queue['segmentation'] = segmentation_list
        if 'instance' in queue[self.queue_length].keys():
            ret_queue['instance'] = instance_list
        if self.turn_on_flow and not self.test_mode:
            ret_queue['flow'] = flow_list


        # 5. Prepare gt_traj, gt_future_boxes, sample_traj for e2e planning.
        future_ego_pos = previous_queue[-1]['sdc_planning']
        future_ego_pos = np.concatenate([np.array([0.,0.,0.])[None], future_ego_pos], axis=0)
        future_ego_pos[:, 2] = future_ego_pos[:, 2] / 180 * np.pi
        sdc_planning = future_ego_pos[1:] - future_ego_pos[:-1]
        sdc_planning_mask = previous_queue[-1]['sdc_planning_mask']
        command = previous_queue[-1]['command']
        # sdc_planning sdc_planning_mask command for planning
        ret_queue['sdc_planning'] = sdc_planning
        ret_queue['sdc_planning_mask'] = sdc_planning_mask
        ret_queue['command'] = command
        # gt_future_boxes segmentation_bev for planning loss and metric
        if 'gt_future_boxes' in previous_queue[-1].keys():
            gt_future_boxes = previous_queue[-1]['gt_future_boxes']
            ret_queue['gt_future_boxes'] = DC(gt_future_boxes, cpu_only=True)
        if 'segmentation_bev' in previous_queue[-1].keys():
            segmentation_bev = np.array(previous_queue[-1]['segmentation_bev'])
            ret_queue['segmentation_bev'] = segmentation_bev
        # sample_traj for traj proposals in planning
        sample_traj = previous_queue[-1]['sample_traj']
        ret_queue['sample_traj'] = sample_traj


        # 6. Prepare gt velocity and steering for action conditions.
        vel_steering_list = np.array([each['vel_steering'] for each in future_queue])
        # caculate vel
        ego_vw = (future_ego_pos[1:, -1] - future_ego_pos[:-1, -1]) / 0.5
        ego_v = np.linalg.norm(future_ego_pos[1:, :2] - future_ego_pos[:-1, :2], axis=-1) / 0.5
        ego_yaw = future_ego_pos[1:, -1] + np.pi/2
        ego_vx, ego_vy = ego_v * np.cos(ego_yaw), ego_v * np.sin(ego_yaw)
        # vel_steering
        vel = np.concatenate([ego_vx[:,None], ego_vy[:,None], ego_vw[:,None]], axis=-1)
        steering = vel_steering_list[:, -1][:,None]
        ret_queue['vel_steering'] = np.concatenate([vel, steering], axis=-1)


        if len(future_can_bus) < 1 + self.future_length:
            return None
        return ret_queue