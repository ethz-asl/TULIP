import torch
from lib.pointnet2.pointnet2_utils import three_nn
from lib.utils.transforms import transformPoints

def registrationMetrics(RtGT, R, t):
    '''Given the extended ground-truth matrix Rt [B,4,4] and the estimated rotation matrix R[B,3,3] and translation vector t[B,1,3] computes the translation and angular error for each sample in batch.
    Returns: rotationError [B] in degrees and translation error [B] in meters.'''

    assert RtGT.size(1) == RtGT.size(2) == 4, 'RtGT mush have shape [B,4,4]'
    assert R.size(1) == R.size(2) == 3, 'R must have shape [B,3,3]'
    assert t.size(1) == 1 and t.size(2) == 3, 't must have shape [B,1,3]'
    assert RtGT.size(0) == R.size(0) == t.size(0), 'batch size must match'

    tgt = RtGT[:,:3,-1].reshape(-1,1,3)
    Rgt = RtGT[:,:3,:3]

    translationError = torch.linalg.norm(t.detach()-tgt, dim=-1).view(-1)
    rotationErrorTrace = (R.detach().transpose(1,2) @ Rgt).diagonal(offset=0, dim1=-1, dim2=-2).reshape(-1,3).sum(dim=1)
    rotationError = torch.rad2deg(torch.acos(torch.clamp(0.5*(rotationErrorTrace-1), -0.999999, 0.999999))).view(-1)

    return rotationError, translationError

def ecdf(x):
    '''Given a tensor, provides ECDF of the values'''
    x = x.reshape(-1)
    n = x.shape[0]
    d = torch.arange(n)/n
    return torch.sort(x)[0].cpu().numpy(), d.numpy()

def overlap_ratio(pts, RtGT, distance_threshold):
    '''Computes overlap ratio between pair of point clouds as percentage of points in source that, when registered,
     are within a distance_threshold distance to any point in the target point cloud.
     Args: pts: torch.FloatTensor shaped [2, npts, 3 or 4],
           RtGt: torch.FloatTensor shaped [4, 4],
           distance_threshold: float'''
    pts = pts.cuda()
    RtGT = RtGT.cuda()

    # get rid of 'padding' points
    pts0, pts1 = pts
    vmask0 = pts0.sum(dim=1) != 0
    vmask1 = pts1.sum(dim=1) != 0
    pts0 = pts0[vmask0, :3]
    pts1 = pts1[vmask1, :3]

    # transform src pts using GT transformation
    pts0_t = transformPoints(RtGT, pts0)

    # compute distance to NN's
    dist, _ = three_nn(pts0_t.unsqueeze(0).contiguous(), pts1.unsqueeze(0).contiguous())
    dist = dist[0, :, 0]

    # compute overlap as ratio of points whose NN's distance is below threshold
    overlap = (dist < distance_threshold).float().mean().item()
    return overlap