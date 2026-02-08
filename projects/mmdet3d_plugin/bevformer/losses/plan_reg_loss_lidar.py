import torch

def plan_reg_loss(pred_pose, rel_poses, gt_modes, return_last=False, loss_type='l2'):
    """
        pred_pose: B,Lout,num_modes=3,2
        rel_poses: B,Lout,2
        gt_modes: B,Lout,num_modes=3
    """
    bs, num_frames, num_modes, _ = pred_pose.shape
    
    pred_pose = pred_pose.transpose(1, 2) # B, M, F, 2
    pred_pose = torch.cumsum(pred_pose, -2)   # B,M,F,2

    gt_modes = gt_modes.transpose(1, 2) # B,M,F
    rel_poses = rel_poses.unsqueeze(1).repeat(1, num_modes, 1, 1)   # B, M(repeat), F, 2
    rel_poses = torch.cumsum(rel_poses, -2) # B,M,F,2

    if return_last:
        pred_pose = pred_pose.new_tensor(pred_pose[:, :, -1:])
        gt_modes = gt_modes.new_tensor(gt_modes[:, :, -1:])
        rel_poses = rel_poses.new_tensor(rel_poses[:, :, -1:])
        bs, num_modes, num_frames, _ = pred_pose.shape
        assert num_frames == 1
        
    if loss_type == 'l1':
        weight = gt_modes[..., None].repeat(1, 1, 1, 2) # B,M,F,2
        loss = torch.abs(pred_pose - rel_poses) * weight
    elif loss_type == 'l2':
        weight = gt_modes # B,M,F
        loss = torch.sqrt(((pred_pose - rel_poses) ** 2).sum(-1)) * weight  # B,M,F
    
    loss = loss.sum() / bs/ num_frames
    return {'loss_reg': loss}