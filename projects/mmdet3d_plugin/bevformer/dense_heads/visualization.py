import os
import torch
import matplotlib.pyplot as plt

def save_bev_render_compare(prev_bev, post_bev,
                            bs, num_frames, bev_w, bev_h, embed_dim,
                            batch_idx=0, frame_idx=0,
                            channels=(0, 1, 2),
                            save_dir='work_dirs/vis_bev',
                            prefix='bev_render'):

    os.makedirs(save_dir, exist_ok=True)
    prev_bt = prev_bev.view(bs, num_frames, bev_w, bev_h, embed_dim)
    post_bt = post_bev.view(bs, num_frames, bev_w, bev_h, embed_dim)

    prev = prev_bt[batch_idx, frame_idx]
    post = post_bt[batch_idx, frame_idx]
    # 过滤掉越界通道
    ch_idx = [c for c in channels if c < embed_dim]
    if len(ch_idx) == 0:
        print('no valid channels, skip')
        return
    # 选通道 -> (W,H,len(ch))
    prev_sel = prev[..., ch_idx]
    post_sel = post[..., ch_idx]
    # 在通道维上取最大值 -> (W,H)
    prev_max = prev_sel.mean(dim=-1)
    post_max = post_sel.mean(dim=-1)
    def norm(x):
        x_min, x_max = x.min(), x.max()
        if (x_max - x_min) < 1e-6:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    prev_img = norm(prev_max).detach().cpu()  # (W,H)
    post_img = norm(post_max).detach().cpu()
    # (W,H) -> (H,W) 方便 imshow
    prev_img = prev_img.T
    post_img = post_img.T
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(prev_img, cmap='viridis')
    plt.title(f'before render, max over ch={ch_idx}')
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(post_img, cmap='viridis')
    plt.title(f'after render, max over ch={ch_idx}')
    plt.axis('off')
    plt.tight_layout()
    fname = f'{prefix}_b{batch_idx}_f{frame_idx}_maxch{"-".join(map(str,ch_idx))}.png'
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'saved to {path}')
