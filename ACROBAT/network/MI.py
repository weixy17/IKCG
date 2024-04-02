import numpy as np

def thirdOrderSplineKernel(u):
    abs_u = np.abs(u)
    sqr_u = abs_u**2.0

    result = np.zeros_like(u)

    mask1 = abs_u<1.0
    mask2 = (abs_u>=1.0 )&(abs_u<2.0)

    result[mask1] = (4.0 - 6.0 * sqr_u[mask1] + 3.0 * sqr_u[mask1] * abs_u[mask1]) / 6.0
    result[mask2] = (8.0 - 12.0 * abs_u[mask2] + 6.0 * sqr_u[mask2] - sqr_u[mask2] * abs_u[mask2]) / 6.0

    return result

def MILoss_wxy(moving, fixed):
    moving = moving.reshape(-1)
    fixed = fixed.reshape(-1)

    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())*255
    moving=(moving-moving.min())/(moving.max()-moving.min())*255
    px=np.histogram(fixed,256,(0,255))[0]/ fixed.shape[-1]
    py=np.histogram(moving,256,(0,255))[0]/ moving.shape[-1]

    
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(fixed, moving, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * fixed.shape[-1])
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    MI = hx + hy - hxy
    
    return MI