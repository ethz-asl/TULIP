import torch
import numpy as np

def getTransform(x, y, z, pitch, yaw, roll, degrees=True, numpy=True):
    '''Given location x,y,z and pitch, yaw, roll, obtain the matrix that convert from local to global CS using the left-handed system from UE4'''

    if degrees:
        pitch, yaw, roll = [np.radians(x) for x in [pitch, yaw, roll]]

    cy,sy = np.cos(yaw), np.sin(yaw)
    cr,sr = np.cos(roll), np.sin(roll)
    cp,sp = np.cos(pitch), np.sin(pitch)

    mat = np.array([cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x, \
                    cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y, \
                         sp,               -cp * sr,                 cp * cr, z, \
                          0,                      0,                       0, 1], dtype=np.float).reshape(4,4)

    if not numpy:
        mat = torch.from_numpy(mat).float()

    return mat

def getRandomTransform(tmean, tsig, rmean, rsig, size=1, numpy=True):
    '''Get pose noise transform with X,Y translation components and rotation along the vertical axis (yaw). Returns Rt [size,4,4] if size>1 or [4,4] if size==1'''

    trs = []
    for _ in range(size):
        t = np.random.normal(loc=tmean, scale=tsig, size=2)
        r = np.random.normal(loc=rmean, scale=rsig, size=1)
        trs.append(getTransform(t[0], t[1], 0, 0, r, 0, degrees=True, numpy=numpy))

    if size == 1:
        return trs[0]
    else:
        if numpy:
            return np.concatenate([tr[None] for tr in trs], axis=0)
        else:
            return torch.cat([tr.unsqueeze(0) for tr in trs], dim=0)

def transformPointsNP(transformMatrix, pts, inverse=False):
    '''Given a transformation matrix [B,4,4] convert pts [B,N,3] or [B,N,4] (last coordinate is intensity)'''
    assert len(pts.shape) == 3, 'pts should have three dimentions: pts [B,N,D], D is 3 or 4'
    assert pts.shape[2] == 3 or pts.shape[2] == 4, 'pts should have shape [B,N,3] or [B,N,4]'
    assert pts.shape[0] == transformMatrix.shape[0], 'pts and matrix should have the same batch size'
    assert type(transformMatrix) == type(pts) == np.ndarray, 'transform and pts must be Numpy arrays '

    #Create homogeneus version of pts
    ptsh = np.concatenate([pts[:,:,:3], np.ones((pts.shape[0],pts.shape[1],1))], axis=2)

    #Obtain intensity, if available
    intensity = pts[:,:,-1:] if pts.shape[2] == 4 else None

    #perform transformation
    mat = transformMatrix
    if inverse:
        mat = np.linalg.inv(mat)
    ptst = ptsh @ mat.transpose((0,2,1))
    ptst = ptst[:,:,:3]

    #merge intensity back
    if intensity is not None:
        ptst = np.concatenate([ptst,intensity], axis=2)

    ptst = ptst.astype(np.float32)
    return ptst 

def transformPointsT(transformMatrix, pts, inverse=False):
    '''Given a transformation matrix [B,4,4] convert pts [B,N,3] or [B,N,4] (last coordinate is intensity)'''
    assert len(pts.shape) == 3, 'pts should have three dimentions: pts [B,N,D], D is 3 or 4'
    assert pts.shape[2] == 3 or pts.shape[2] == 4, 'pts should have shape [B,N,3] or [B,N,4]'
    assert pts.shape[0] == transformMatrix.shape[0], 'pts and matrix should have the same batch size'
    assert type(transformMatrix) == type(pts) == torch.Tensor, 'transform and pts must be torch tensors'
    assert transformMatrix.device == pts.device, 'transform and pts must be on the same device'

    #Create homogeneus version of pts
    ptsh = torch.cat([pts[:,:,:3], torch.ones_like(pts[:,:,0:1])], dim=2)

    #Obtain intensity, if available
    intensity = pts[:,:,-1:] if pts.shape[2] == 4 else None

    #perform transformation
    mat = transformMatrix
    if inverse:
        mat = torch.inverse(mat)
    ptst = ptsh @ mat.transpose(1,2)
    ptst = ptst[:,:,:3]

    #merge intensity back
    if intensity is not None:
        ptst = torch.cat([ptst,intensity], dim=2)

    return ptst 


def transformPoints(transformMatrix, pts, inverse=False):
    '''Given a transformation matrix [4,4] convert pts [N,3] or [N,4] (last coordinate is intensity)'''

    assert len(pts.shape) == 2, 'pts should have two dimentions'
    assert pts.shape[1] == 3 or pts.shape[1] == 4, 'pts should have shape [N,3] or [N,4]'
    assert type(transformMatrix) == type(pts) , 'transform and pts must be the same type'
    assert type(transformMatrix) == np.ndarray or type(transformMatrix) == torch.Tensor , 'transform and pts be either Numpy arrays or Torch tensors'

    if type(transformMatrix) == np.ndarray:
        return transformPointsNP(transformMatrix[None], pts[None], inverse)[0]
    else:
        return transformPointsT(transformMatrix.unsqueeze(0), pts.unsqueeze(0), inverse)[0]

def transformPointsBatch(transformMatrix, pts, inverse=False):
    '''Given a transformation matrix [B,4,4] convert pts [B,N,3] or [B,N,4] (last coordinate is intensity)'''

    assert len(pts.shape) == 3, 'pts should have three dimentions'
    assert pts.shape[2] == 3 or pts.shape[2] == 4, 'pts should have shape [B,N,3] or [B,N,4]'
    assert pts.shape[0] == transformMatrix.shape[0], 'pts and matrix should have the same batch size'
    assert type(transformMatrix) == type(pts) , 'transform and pts must be the same type'
    assert type(transformMatrix) == np.ndarray or type(transformMatrix) == torch.Tensor , 'transform and pts be either Numpy arrays or Torch tensors'

    if type(transformMatrix) == np.ndarray:
        return transformPointsNP(transformMatrix, pts, inverse)
    else:
        return transformPointsT(transformMatrix, pts, inverse)
