import mmcv
import numpy as np
import os

nus_root_path = 'data/nuscnes'

nus_info_train = 'nuscenes_occ_infos_train.pkl' # Cam4dOCC
nus_info_val = 'nuscenes_occ_infos_val.pkl'

occworld_info_train = 'nuscenes_infos_train_temporal_v3_scene.pkl'  # OccWorld
occworld_info_val = 'nuscenes_infos_val_temporal_v3_scene.pkl'

vad_info_train = 'vad_nuscenes_infos_temporal_train.pkl'    # VAD
vad_info_val = 'vad_nuscenes_infos_temporal_val.pkl'

# anno_train
nus_anno = mmcv.load(nus_info_train)
occ_anno = mmcv.load(occworld_info_train)
vad_anno = mmcv.load(vad_info_train)

i = 0
new_info_train = []
for scene_id, scene_info in occ_anno['infos'].items():
    for frame_info in scene_info:
        nus_info = nus_anno['infos'][i]
        vad_info = vad_anno['infos'][i]
        print(nus_info['token'])
        assert nus_info['token'] == frame_info['token']
        assert nus_info['token'] == vad_info['token']
        nus_info.update({
            'gt_ego_fut_trajs': frame_info['gt_ego_fut_trajs'],
            'pose_mode': frame_info['pose_mode'],
            'vel_steering': np.array([vad_info['gt_ego_lcf_feat'][0],vad_info['gt_ego_lcf_feat'][1],    # vx,vy(m/s)
                                        vad_info['gt_ego_lcf_feat'][4],vad_info['gt_ego_lcf_feat'][8]]),     # v_yaw(rad/s), steering
        })
        new_info_train.append(nus_info)
        i += 1

new_anno_train = {
    'infos': new_info_train,
    'metadata': nus_anno['metadata']
}
mmcv.dump(new_anno_train, os.path.join(nus_root_path, 'nuscenes_infos_temporal_train_new.pkl'))


# anno_val
nus_anno = mmcv.load(nus_info_val)
occ_anno = mmcv.load(occworld_info_val)
vad_anno = mmcv.load(vad_info_val)

i = 0
new_info_val = []
for scene_id, scene_info in occ_anno['infos'].items():
    for frame_info in scene_info:
        nus_info = nus_anno['infos'][i]
        vad_info = vad_anno['infos'][i]
        print(nus_info['token'])
        assert nus_info['token'] == frame_info['token']
        assert nus_info['token'] == vad_info['token']
        nus_info.update({
            'gt_ego_fut_trajs': frame_info['gt_ego_fut_trajs'],
            'pose_mode': frame_info['pose_mode'],
            'vel_steering': np.array([vad_info['gt_ego_lcf_feat'][0],vad_info['gt_ego_lcf_feat'][1],    # vx,vy(m/s)
                                        vad_info['gt_ego_lcf_feat'][4],vad_info['gt_ego_lcf_feat'][8]]),     # v_yaw(rad/s), steering
        })
        new_info_val.append(nus_info)
        i += 1

new_anno_val = {
    'infos': new_info_val,
    'metadata': nus_anno['metadata']
}
mmcv.dump(new_anno_val, os.path.join(nus_root_path, 'nuscenes_infos_temporal_val_new.pkl'))

