import copy
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from .nuscenes_dataset import CustomNuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
import numpy as np
import cv2
import torch
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.formating import cm_to_ious, format_iou_results
from projects.mmdet3d_plugin.datasets.trajectory_api import NuScenesTraj
from projects.mmdet3d_plugin.datasets.samplers import sampler as trajectory_sampler
from projects.mmdet3d_plugin.bevformer.dense_heads.plan_head import calculate_birds_eye_view_parameters
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from prettytable import PrettyTable


@DATASETS.register_module()
class NuScenesWorldDatasetTemplate(CustomNuScenesDataset):
    r"""World dataset for visual point cloud forecasting.
    """

    def __init__(self,
                 classes,
                 use_separate_classes,
                 use_fine_occ,
                 turn_on_flow,
                 future_length,
                 ego_mask=None,
                 load_frame_interval=None,
                 rand_frame_interval=(1,),
                 plan_grid_conf=None,

                 *args,
                 **kwargs):
        """
        Args:
            future_length: the number of predicted future point clouds.
            ego_mask: mask points belonging to the ego vehicle.
            load_frame_interval: partial of training set.
            rand_frame_interval: augmentation for future prediction.
        """
        # Hack the original {self._set_group_flag} function.
        self.usable_index = []

        super().__init__(*args, **kwargs)
        self.classes = classes
        self.use_separate_classes = use_separate_classes
        self.use_fine_occ = use_fine_occ
        self.turn_on_flow = turn_on_flow
        # load origin nusc dataset for instance annotation
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=False)
        self.nusc_can = NuScenesCanBus(dataroot=self.data_root)

        # scene2map
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
        
        # traj_api
        self.traj_api = NuScenesTraj(self.nusc,
                                     self.CLASSES,
                                     self.box_mode_3d,
                                     planning_steps=future_length+1)
        
        # ignore_label_name
        self.ignore_bbox_label_name = ['barrier', 'traffic_cone', 'animal', 'noise',
                                       'movable_object.debris', 'movable_object.pushable_pullable', 'static_object.bicycle_rack']

        self.plan_grid_conf = plan_grid_conf
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            plan_grid_conf['xbound'], plan_grid_conf['ybound'], plan_grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()   # [0.5 0.5 20]
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()     # [200 200 1]

        self.future_length = future_length  # 2
        self.ego_mask = ego_mask            # (-0.8, -1.5, 0.8, 2.5)
        self.load_frame_interval = load_frame_interval  # 8
        self.rand_frame_interval = rand_frame_interval  # (-1, 1)

        # Remove data_infos without enough history & future.
        # if test, assert all history frames are available
        #  Align with the setting of 4D-occ: https://github.com/tarashakhurana/4d-occ-forecasting
        last_scene_index = None
        last_scene_frame = -1
        usable_index = []
        # valid_prev_length = (self.queue_length if self.test_mode else 0)
        valid_prev_length = self.queue_length
        for index, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            if last_scene_index != info['scene_token']:
                last_scene_index = info['scene_token']
                last_scene_frame = -1
            last_scene_frame += 1
            if last_scene_frame >= valid_prev_length:
                # has enough previous frame.
                # now, let's check whether it has enough future frame.
                tgt_future_index = index + self.future_length
                if tgt_future_index >= len(self.data_infos):
                    break
                if last_scene_index != self.data_infos[tgt_future_index]['scene_token']:
                    # the future scene is not corresponded to the current scene
                    continue
                usable_index.append(index)

        # Remove useless frame index if load_frame_interval is assigned.
        if self.load_frame_interval is not None:
            usable_index = usable_index[::self.load_frame_interval]
        self.usable_index = usable_index

        if not self.test_mode:
            self._set_group_flag()
    
    def reframe_boxes(self, boxes, t_init, t_curr):
        l2e_r_mat_curr = t_curr['l2e_r']
        l2e_t_curr = t_curr['l2e_t']
        e2g_r_mat_curr = t_curr['e2g_r']
        e2g_t_curr = t_curr['e2g_t']

        l2e_r_mat_init = t_init['l2e_r']
        l2e_t_init = t_init['l2e_t']
        e2g_r_mat_init = t_init['e2g_r']
        e2g_t_init = t_init['e2g_t']

        # to bbox under curr ego frame  # TODO: Uncomment
        boxes.rotate(l2e_r_mat_curr.T)
        boxes.translate(l2e_t_curr)

        # to bbox under world frame
        boxes.rotate(e2g_r_mat_curr.T)
        boxes.translate(e2g_t_curr)

        # to bbox under initial ego frame, first inverse translate, then inverse rotate 
        boxes.translate(- e2g_t_init)
        m1 = np.linalg.inv(e2g_r_mat_init)
        boxes.rotate(m1.T)

        # to bbox under curr ego frame, first inverse translate, then inverse rotate
        boxes.translate(- l2e_t_init)
        m2 = np.linalg.inv(l2e_r_mat_init)
        boxes.rotate(m2.T)

        return boxes
    
    def get_future_bboxes(self, index):
        cur_info = self.data_infos[index]

        # ref pose
        dtype = torch.float32
        l2e_r = cur_info['lidar2ego_rotation']
        l2e_t = cur_info['lidar2ego_translation']
        e2g_r = cur_info['ego2global_rotation']
        e2g_t = cur_info['ego2global_translation']
        l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix).to(dtype)
        e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix).to(dtype)
        l2e_t_vec = torch.tensor(l2e_t).to(dtype)
        e2g_t_vec = torch.tensor(e2g_t).to(dtype)
        t_ref = dict(l2e_r=l2e_r_mat, l2e_t=l2e_t_vec, e2g_r=e2g_r_mat, e2g_t=e2g_t_vec)

        segmentations = []
        gt_future_boxes = []

        # generate the future
        index_list = list(range(index + 1, index + (self.future_length + 2)))

        for fur_index in index_list:
            if fur_index < len(self.data_infos) and self.data_infos[fur_index]['scene_token'] == cur_info['scene_token']:
                fur_info = self.data_infos[fur_index]
                # future_gt_bbox
                gt_bboxes_3d = fur_info['gt_boxes'].copy()
                # not exist gt_bbox
                if gt_bboxes_3d.shape[0] == 0:
                    segmentation = np.zeros(
                        (self.bev_dimension[1], self.bev_dimension[0]))
                else:
                    gt_velocity = fur_info['gt_velocity']
                    nan_mask = np.isnan(gt_velocity[:, 0])
                    gt_velocity[nan_mask] = [0.0, 0.0]
                    gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
                    gt_bboxes_3d = LiDARInstance3DBoxes(
                        gt_bboxes_3d,
                        box_dim=gt_bboxes_3d.shape[-1],
                        origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
                    # future pose
                    l2e_r = fur_info['lidar2ego_rotation']
                    l2e_t = fur_info['lidar2ego_translation']
                    e2g_r = fur_info['ego2global_rotation']
                    e2g_t = fur_info['ego2global_translation']
                    l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix).to(dtype)
                    e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix).to(dtype)
                    l2e_t_vec = torch.tensor(l2e_t).to(dtype)
                    e2g_t_vec = torch.tensor(e2g_t).to(dtype)
                    t_curr = dict(l2e_r=l2e_r_mat, l2e_t=l2e_t_vec, e2g_r=e2g_r_mat, e2g_t=e2g_t_vec)
                    # reframe bboxes
                    gt_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr)  # N_box

                    # segmentation
                    segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))
                    # select box
                    gt_bboxes_names = fur_info['gt_names'].copy().tolist()
                    select_mask = [name not in self.ignore_bbox_label_name for name in gt_bboxes_names]
                    select_gt_bboxes = gt_bboxes_3d[select_mask]
                    # valid sample andd has objects
                    if len(select_gt_bboxes.tensor) > 0:
                        bbox_corners = select_gt_bboxes.corners[:, [
                            0, 3, 7, 4], :2].numpy()
                        bbox_corners = bbox_corners[..., [1, 0]]    # NOTE: H:lidar_x  W:lidar_y
                        bbox_corners = np.round(
                            (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
                        # plot segmentation
                        for poly_region in bbox_corners:
                            cv2.fillPoly(segmentation, [poly_region], 1.0)
            else:
                gt_bboxes_3d = None
                segmentation = np.zeros(
                        (self.bev_dimension[1], self.bev_dimension[0])) # H,W = ignore
            
            gt_future_boxes.append(gt_bboxes_3d)
            segmentations.append(segmentation)

        return gt_future_boxes, segmentations

    def get_trajectory_sampling(self, rec, future_length, SAMPLE_INTERVAL=0.5):
        try:
            ref_scene = self.nusc.get("scene", rec['scene_token'])

            # vm_msgs = self.nusc_can.get_messages(ref_scene['name'], 'vehicle_monitor')
            # vm_uts = [msg['utime'] for msg in vm_msgs]
            pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')    # 167
            pose_uts = [msg['utime'] for msg in pose_msgs]
            steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
            steer_uts = [msg['utime'] for msg in steer_msgs]

            ref_utime = rec['timestamp']
            # vm_index = locate_message(vm_uts, ref_utime)
            # vm_data = vm_msgs[vm_index]
            pose_index = trajectory_sampler.locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = trajectory_sampler.locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]

            # initial speed
            # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
            v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s

            # curvature (positive: turn left)
            # steering = np.deg2rad(vm_data["steering"])
            steering = steer_data["value"]

            location = self.scene2map[ref_scene['name']]
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if location.startswith('singapore') else False
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588
        except: # self.nusc_can.can_blacklist: some scenes does not have vehicle monitor data
            v0 = 6.6
            Kappa = 0

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = future_length * SAMPLE_INTERVAL  # second
        t_interval = SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, 1800)  # [720, 360, 720] Left,Straight,Right
        sampled_trajectories = sampled_trajectories_fine[:, ::10]   # sample_num, start+future, 3
        sampled_trajectories = sampled_trajectories[:, 1:] - sampled_trajectories[:, :-1]  # sample_num, future, 3
        return sampled_trajectories

    def get_data_info(self, index):
        """Also return lidar2ego transformations."""
        input_dict = super().get_data_info(index)

        info = self.data_infos[index]

        input_dict.update(dict(
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            cam2img=input_dict['cam_intrinsic'],
            lidar_token=info['lidar_token'],
            vel_steering=info['vel_steering'],        # 1,4   vx(m/s),vy(m/s),v_yaw(rad/s),steering
        ))
        return input_dict
    
    def get_lidar_pose(self, rec):
        '''
        Get global poses for following bbox transforming
        '''
        ego2global_translation = rec['ego2global_translation']
        ego2global_rotation = rec['ego2global_rotation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        
        return trans, rot
    
    def get_ego2lidar_pose(self, rec):
        '''
        Get LiDAR poses in ego system
        '''
        lidar2ego_translation = rec['lidar2ego_translation']
        lidar2ego_rotation = rec['lidar2ego_rotation']
        trans = -np.array(lidar2ego_translation)
        rot = Quaternion(lidar2ego_rotation).inverse
        return trans, rot
    
    def record_instance(self, idx, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it
        """
        rec = self.data_infos[idx]
        self.scene_token.append(rec['scene_token'])
        self.lidar_token.append(rec['lidar_token'])
        translation, rotation = self.get_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])
        ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
        self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

        current_sample = self.nusc.get('sample', rec['token'])
        for annotation_token in current_sample['anns']:
            annotation = self.nusc.get('sample_annotation', annotation_token)
            # Instance extraction for Cam4DOcc-V1 
            # Filter out all non vehicle instances
            # if 'vehicle' not in annotation['category_name']:
            #     continue
            gmo_flag = False
            for class_name in self.classes:
                if class_name in annotation['category_name']:
                    gmo_flag = True
                    break
            if not gmo_flag:
                continue
            # Specify semantic id if use_separate_classes
            semantic_id = 1
            if self.use_separate_classes:
                if 'bicycle' in annotation['category_name']:
                    semantic_id = 1
                elif 'bus'  in annotation['category_name']:
                    semantic_id = 2
                elif 'car'  in annotation['category_name']:
                    semantic_id = 3
                elif 'construction'  in annotation['category_name']:
                    semantic_id = 4
                elif 'motorcycle'  in annotation['category_name']:
                    semantic_id = 5
                elif 'trailer'  in annotation['category_name']:
                    semantic_id = 6
                elif 'truck'  in annotation['category_name']:
                    semantic_id = 7
                elif 'pedestrian'  in annotation['category_name']:
                    semantic_id = 8

            # Filter out invisible vehicles
            FILTER_INVISIBLE_VEHICLES = True
            if FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in self.visible_instance_set:
                continue
            # Filter out vehicles that have not been seen in the past
            if self.counter >= (self.queue_length+1) and annotation['instance_token'] not in self.visible_instance_set:
                continue
            self.visible_instance_set.add(annotation['instance_token'])

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1  # instance_map={'instance_token': instance_id}
            instance_id = instance_map[annotation['instance_token']]
            instance_attribute = int(annotation['visibility_token'])

            if annotation['instance_token'] not in self.instance_dict:
                # For the first occurrence of an instance
                self.instance_dict[annotation['instance_token']] = {
                    'timestep': [self.counter],
                    'translation': [annotation['translation']],
                    'rotation': [annotation['rotation']],
                    'size': annotation['size'],
                    'instance_id': instance_id,
                    'semantic_id': semantic_id,
                    'attribute_label': [instance_attribute],
                }
            else:
                # For the instance that have appeared before
                self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
                self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map
    
    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True

    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.queue_length+1+self.future_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue
            
            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1
        
        return instance

    def _prepare_data_info_single(self, index, occ_load_flag=None, aug_param=None):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        if aug_param is not None:
            input_dict['aug_param'] = copy.deepcopy(aug_param)
        
        # only load current frame
        if occ_load_flag is not None:
            input_dict['occ_load_flag'] = occ_load_flag
        # gt_future_boxes for planning loss and metric
        if occ_load_flag:   # only load current frame
            gt_future_boxes, segmentation_bev = self.get_future_bboxes(index)
            input_dict.update(
                gt_future_boxes=gt_future_boxes,
                segmentation_bev=segmentation_bev,
            )
        # sdc_planning for planning loss and metric
        if occ_load_flag:
            # sdc_plan
            info = self.data_infos[index]
            sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(info['token'])
            sample_traj = self.get_trajectory_sampling(info, self.future_length+1)
            input_dict.update(
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                sample_traj=sample_traj,
            )
        # prepare instance dict
        if occ_load_flag:    # only load current frame
            cur_index_list = list(range(index-self.queue_length, index + (self.future_length + 1)))
            self.scene_token = []
            self.lidar_token = []
            self.egopose_list = []
            self.ego2lidar_list = []
            self.visible_instance_set = set()
            self.instance_dict = {}
            instance_map = {}
            # load annotation to instance_dict
            for self.counter, index_t in enumerate(cur_index_list):
                instance_map = self.record_instance(index_t, instance_map)
            # fix missing
            for token in self.instance_dict.keys():
                self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])
            # update input_dict
            input_dict.update(
                use_fine_occ=self.use_fine_occ,
                scene_token_list=self.scene_token,
                lidar_token_list=self.lidar_token,
                egopose_list=self.egopose_list,
                ego2lidar_list=self.ego2lidar_list,
                instance_dict=self.instance_dict,
                instance_map=instance_map,
            )

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def _prepare_data_info(self, index, rand_interval=None):
        """
        Modified from BEVFormer:CustomNuScenesDataset,
            BEVFormer logits: randomly select (queue_length-1) previous images.
            Modified logits: directly select (queue_length) previous images.
        """
        rand_interval = (
            rand_interval if rand_interval is not None else
            np.random.choice(self.rand_frame_interval, 1)[0]
        )

        # 1. get previous camera information.
        previous_queue = [] # history_len*['img', 'points', 'aug_param']
        previous_index_list = list(range(
            index - self.queue_length * rand_interval, index, rand_interval))
        previous_index_list = sorted(previous_index_list)
        if rand_interval < 0:  # the inverse chain.
            previous_index_list = previous_index_list[::-1]
        previous_index_list.append(index) 
        aug_param = None
        for i, idx in enumerate(previous_index_list):
            idx = min(max(0, idx), len(self.data_infos) - 1)

            occ_load_flag = True if i==self.queue_length else False     # only load current frame

            example = self._prepare_data_info_single(idx, occ_load_flag, aug_param=aug_param)

            aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else None
            if example is None:
                return None
            previous_queue.append(example)

        # 2. get future occ information.
        future_queue = []
        # Future: from current to future frames.
        # use current frame as the 0-th future.
        future_index_list = list(range(
            index, index + (self.future_length + 1) * rand_interval, rand_interval))
        future_index_list = sorted(future_index_list)
        if rand_interval < 0:  # the inverse chain.
            future_index_list = future_index_list[::-1]
        has_future = False
        for i, idx in enumerate(future_index_list):
            idx = min(max(0, idx), len(self.data_infos) - 1)

            occ_load_flag = False

            example = self._prepare_data_info_single(idx, occ_load_flag, aug_param=aug_param)

            aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else None
            if example is None and not has_future:
                return None
            future_queue.append(example)
            has_future = True
        return self.union2one(previous_queue, future_queue)

    def union2one(self, previous_queue, future_queue):
        pass

    def evaluate(self, results, logger=None, **kawrgs):
        '''
        Evaluate by IOU and VPQ metrics for model evaluation
        '''
        eval_results = {}
        
        ''' calculate IOU of current and future frames'''
        if 'hist_for_iou' in results.keys():
            IoU_results_current_future = {}
            hist_for_iou = sum(results['hist_for_iou'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_current_future['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of current and future frames:')
                logger.info(res_table)        
            eval_results.update(IoU_of_Current_Future=IoU_results_current_future)
        
        ''' calculate IOU of current frame'''
        if 'hist_for_iou_current' in results.keys():
            IoU_results_current = {}
            hist_for_iou = sum(results['hist_for_iou_current'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_current['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of current frame:')
                logger.info(res_table)        
            eval_results.update(IoU_of_Current=IoU_results_current)
        
        ''' calculate IOU of future frame'''
        if 'hist_for_iou_future' in results.keys():
            IoU_results_future = {}
            hist_for_iou = sum(results['hist_for_iou_future'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_future['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of future frames:')
                logger.info(res_table)        
            eval_results.update(IoU_of_Future=IoU_results_future)
        
        ''' calculate IOU of future frame with time_weighting'''
        if 'hist_for_iou_future_time_weighting' in results.keys():
            IoU_results_future_time_weighting = {}
            hist_for_iou = sum(results['hist_for_iou_future_time_weighting'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_future_time_weighting['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of future frames with time weighting:')
                logger.info(res_table)        
            eval_results.update(IoU_of_Future_with_Time_Weighting=IoU_results_future_time_weighting)

        ''' calculate plan_metric '''
        if 'plan_metric' in results.keys() and 'data_len' in results.keys():
            for key, value in results['plan_metric'].items():
                eval_results[key] = sum(value)/results['data_len']
            eval_results.update(avg_l2=(eval_results['plan_L2_1s']+eval_results['plan_L2_2s']+eval_results['plan_L2_3s'])/3)
            eval_results.update(avg_obj_col=(eval_results['plan_obj_col_1s']+eval_results['plan_obj_col_2s']+eval_results['plan_obj_col_3s'])/3)
            eval_results.update(avg_obj_box_col=(eval_results['plan_obj_box_col_1s']+eval_results['plan_obj_box_col_2s']+eval_results['plan_obj_box_col_3s'])/3)
            eval_results.update(avg_obj_box_col_single=(eval_results['plan_obj_box_col_1s_single']+eval_results['plan_obj_box_col_2s_single']+eval_results['plan_obj_box_col_3s_single'])/3)
            eval_results.update(avg_obj_col_single=(eval_results['plan_obj_col_1s_single']+eval_results['plan_obj_col_2s_single']+eval_results['plan_obj_col_3s_single'])/3)
            eval_results.update(avg_l2_single=(eval_results['plan_L2_1s_single']+eval_results['plan_L2_2s_single']+eval_results['plan_L2_3s_single'])/3)
        
        if 'planning_results_computed' in results.keys():
            planning_results_computed = results['planning_results_computed']
            num_frames = len(planning_results_computed['L2'])
            planning_tab = PrettyTable()
            planning_tab.field_names = [
                "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"][:num_frames+1]
            for key in planning_results_computed.keys():
                value = planning_results_computed[key]  # Lout
                row_value = []
                row_value.append(key)
                for i in range(len(value)):
                    row_value.append('%.4f' % float(value[i]))
                planning_tab.add_row(row_value)
            print(planning_tab)

        return eval_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        rand_interval = None
        while True:
            data = self._prepare_data_info(
                self.usable_index[idx], rand_interval=rand_interval)
            if data is None:
                if self.test_mode:
                    idx += 1
                else:
                    if rand_interval is None:
                        rand_interval = 1  # use rand_interval = 1 for the same sample again.
                    else:  # still None for rand_interval = 1, no enough future.
                        idx = self._rand_another(idx)
                        rand_interval = None
                continue
            assert data is not None
            return data

    def __len__(self):
        return len(self.usable_index)
