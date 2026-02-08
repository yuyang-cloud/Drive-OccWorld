import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.draw import polygon

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class Cost_Function(nn.Module):
    def __init__(self, cfg):
        super(Cost_Function, self).__init__()

        self.safetycost = SafetyCost(cfg)
        self.headwaycost = HeadwayCost(cfg)
        # self.lrdividercost = LR_divider(cfg)
        # self.comfortcost = Comfort(cfg)
        # self.progresscost = Progress(cfg)
        self.rulecost = Rule(cfg)
        self.costvolume = Cost_Volume(cfg)

    def forward(self, cost_volume, trajs, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor (B, N, 2)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)   instance_occupied=1
        drivable_area: torch.Tensor(B, 200, 200)        driveable_surface=1
        '''
        safetycost = torch.clamp(self.safetycost(trajs, instance_occupancy), 0, 100)                 # penalize overlap with instance_occupancy
        headwaycost = torch.clamp(self.headwaycost(trajs, instance_occupancy, drivable_area), 0, 100)# penalize overlap with front instance (10m)
        # lrdividercost = torch.clamp(self.lrdividercost(trajs, lane_divider), 0, 100)               # penalize distance with lane
        # comfortcost = torch.clamp(self.comfortcost(trajs), 0, 100)                                   # penalize high accelerations (lateral, longitudinal, jerk)
        # progresscost = torch.clamp(self.progresscost(trajs), -100, 100)                              # L2 loss
        rulecost = torch.clamp(self.rulecost(trajs, drivable_area), 0, 100)                          # penalize overlap with out of drivable_area
        costvolume = torch.clamp(self.costvolume(trajs, cost_volume), 0, 100)                        # sample on costvolume

        cost_fo = safetycost + headwaycost + costvolume + rulecost
        # cost_fc = progresscost

        return cost_fo



class BaseCost(nn.Module):
    def __init__(self, grid_conf):
        super(BaseCost, self).__init__()
        self.grid_conf = grid_conf

        dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx,requires_grad=False)
        self.bx = nn.Parameter(bx,requires_grad=False)

        _,_, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        )

        self.W = 1.85
        self.H = 4.084

    def get_origin_points(self, lambda_=0):
        W = self.W
        H = self.H
        pts = np.array([
            [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
            [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
        ])  # [lidar_y, lidar_x]
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())   # [bev_w, bev_h]
        # pts[:, [0, 1]] = pts[:, [1, 0]] # [bev_h, bev_w]
        rr , cc = polygon(pts[:,1], pts[:,0])   # [bev_h, bev_w]
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)  # [bev_h, bev_w]
        return torch.from_numpy(rc).to(device=self.bx.device) # (27,2)

    def get_points(self, trajs, lambda_=0):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        return:
        List[ torch.Tensor<int> (B, N), torch.Tensor<int> (B, N)]
        '''
        rc = self.get_origin_points(lambda_)    # [bev_h, bev_w]
        B, N, _ = trajs.shape         # delta_[lidar_x, lidar_y]

        trajs = trajs.view(B, N, 1, 2) / self.dx  # delta_[bev_h, bev_w]
        # trajs[:,:,:,:,[0,1]] = trajs[:,:,:,:,[1,0]]
        trajs = trajs + rc  # [bev_h, bev_w]

        rr = trajs[:,:,:,0].long()
        rr = torch.clamp(rr, 0, self.bev_dimension[0] - 1)

        cc = trajs[:,:,:,1].long()
        cc = torch.clamp(cc, 0, self.bev_dimension[1] - 1)

        return rr, cc

    def compute_area(self, instance_occupancy, trajs, ego_velocity=None, _lambda=0):
        '''
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)
        ego_velocity: torch.Tensor<float> (B, N)
        '''
        _lambda = int(_lambda / self.dx[0])
        rr, cc = self.get_points(trajs, _lambda)    # [bev_h, bev_w]
        B, N, _ = trajs.shape

        if ego_velocity is None:
            ego_velocity = torch.ones((B,N), device=trajs.device)

        ii = torch.arange(B)

        subcost = instance_occupancy[ii[:, None, None], rr, cc].sum(dim=-1)
        subcost = subcost * ego_velocity

        return subcost

    def discretize(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''
        B, N,  _ = trajs.shape # delta_[lidar_x, lidar_y]

        xx, yy = trajs[:,:,0], trajs[:,:,1] # delta_[lidar_x, lidar_y]

        # discretize
        xi = ((xx - self.bx[0]) / self.dx[0]).long()
        xi = torch.clamp(xi, 0, self.bev_dimension[0]-1)    # bev_h

        yi = ((yy - self.bx[1]) / self.dx[1]).long()
        yi = torch.clamp(yi,0, self.bev_dimension[1]-1)     # bev_w

        return xi, yi

    def evaluate(self, trajs, C):
        '''
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            C: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape

        ii = torch.arange(B)

        Syi, Sxi = self.discretize(trajs)

        CS = C[ii, Syi, Sxi]
        return CS

class Cost_Volume(BaseCost):
    def __init__(self, cfg):
        super(Cost_Volume, self).__init__(cfg)

        self.factor = 100.

    def forward(self, trajs, cost_volume):
        '''
        cost_volume: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''

        cost_volume = torch.clamp(cost_volume, 0, 1000)

        return self.evaluate(trajs, cost_volume) * self.factor

class Rule(BaseCost):
    def __init__(self, cfg):
        super(Rule, self).__init__(cfg)

        self.factor = 5

    def forward(self, trajs, drivable_area):
        '''
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, _,  _ = trajs.shape

        dangerous_area = torch.logical_not(drivable_area).float()
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(dangerous_area[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        subcost = self.compute_area(dangerous_area, trajs)

        return subcost * self.factor


class SafetyCost(BaseCost):
    def __init__(self, cfg):
        super(SafetyCost, self).__init__(cfg)
        self.w = nn.Parameter(torch.tensor([1.,1.]),requires_grad=False)

        self._lambda = 1.
        self.factor = 0.1

    def forward(self, trajs, instance_occupancy):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape
        ego_velocity = torch.sqrt((trajs ** 2).sum(axis=-1)) / 0.5  # B,N

        # o_c(tau, t, 0)
        subcost1 = self.compute_area(instance_occupancy, trajs)
        # o_c(tau, t, lambda) x v(tau, t)
        subcost2 = self.compute_area(instance_occupancy, trajs, ego_velocity, self._lambda)

        subcost = subcost1 * self.w[0] + subcost2 * self.w[1]

        return subcost * self.factor


class HeadwayCost(BaseCost):
    def __init__(self, cfg):
        super(HeadwayCost, self).__init__(cfg)
        self.L = 10  # Longitudinal distance keep 10m
        self.factor = 1.

    def forward(self, trajs, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape
        instance_occupancy_ = instance_occupancy * drivable_area  # B,H,W
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(instance_occupancy_[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        tmp_trajs = trajs.clone()
        tmp_trajs[:,:,1] = tmp_trajs[:,:,1]+self.L

        subcost = self.compute_area(instance_occupancy_, tmp_trajs)

        return subcost * self.factor

class LR_divider(BaseCost):
    def __init__(self, cfg):
        super(LR_divider, self).__init__(cfg)
        self.L = 1 # Keep a distance of 2m from the lane line
        self.factor = 10.

    def forward(self, trajs, lane_divider):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        lane_divider: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape

        xx, yy = self.discretize(trajs) # [bev_h, bev_w]
        xy = torch.stack([xx,yy],dim=-1) # (B, N, 2)  [bev_h, bev_w]

        # lane divider
        res1 = []
        for i in range(B):
            index = torch.nonzero(lane_divider[i]) # (n, 2)
            if len(index) != 0:
                xy_batch = xy[i].view(N, 1, 2)
                distance = torch.sqrt((((xy_batch - index) * reversed(self.dx))**2).sum(dim=-1)) # (N, n)
                distance,_ = distance.min(dim=-1) # (N)
                index = distance > self.L
                distance = (self.L - distance) ** 2
                distance[index] = 0
            else:
                distance = torch.zeros((N),device=trajs.device)
            res1.append(distance)
        res1 = torch.stack(res1, dim=0)

        return res1 * self.factor


class Comfort(BaseCost):
    def __init__(self, cfg):
        super(Comfort, self).__init__(cfg)

        self.c_lat_acc = 3 # m/s2
        self.c_lon_acc = 3 # m/s2
        self.c_jerk = 1 # m/s3

        self.factor = 0.1

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        '''
        B, N, _ = trajs.shape
        lateral_velocity = trajs[:,:,0] / 0.5
        longitudinal_velocity = trajs[:,:,1] / 0.5
        lateral_acc = lateral_velocity / 0.5    # B,N
        longitudinal_acc = longitudinal_velocity / 0.5  # B,N

        # jerk
        ego_velocity = torch.sqrt((trajs ** 2).sum(dim=-1)) / 0.5
        ego_acc = ego_velocity / 0.5
        ego_jerk = ego_acc / 0.5    # B,N

        subcost = torch.zeros((B, N), device=trajs.device)

        lateral_acc = torch.clamp(torch.abs(lateral_acc) - self.c_lat_acc, 0,30)
        subcost += lateral_acc ** 2
        longitudinal_acc = torch.clamp(torch.abs(longitudinal_acc) - self.c_lon_acc, 0, 30)
        subcost += longitudinal_acc ** 2
        ego_jerk = torch.clamp(torch.abs(ego_jerk) - self.c_jerk, 0, 20)
        subcost += ego_jerk ** 2

        return subcost * self.factor

class Progress(BaseCost):
    def __init__(self, cfg):
        super(Progress, self).__init__(cfg)
        self.factor = 0.5

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        target_points: torch.Tensor<float> (B, 2)
        '''
        target_points = torch.zeros_like(trajs[:, 0, :])    # B,2
        B, N,  _ = trajs.shape
        subcost1 = trajs[:,:,1]

        if target_points.sum() < 0.5:
            subcost2 = 0
        else:
            target_points = target_points.unsqueeze(1)
            subcost2 = ((trajs - target_points) ** 2).sum(dim=-1)

        return (subcost2 - subcost1) * self.factor
