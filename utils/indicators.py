import numpy as np

def HitRate(pred, true, threshold):
    """ 命中率(Hit rate) """
    if threshold == 'auto':  # 若不指定阈值，则用绝对差值最大值的一半作为阈值
        threshold = np.max(np.abs(pred - true)) * 0.5

    res = np.mean(np.where(abs(pred - true) < threshold, 1, 0))
    return res

