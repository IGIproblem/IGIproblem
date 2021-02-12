import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_rand_score

def miou(y,pred):
    Is = np.empty((1, 2))
    Us = np.empty((1, 2))
    for cl in range(2):
        cur_gt_mask = (y == cl)
        cur_pred_mask = (pred == cl)
        I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
        U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
        Is[0, cl] = I
        Us[0, cl] = U

    ious = np.divide(Is, Us)
    ious[np.isnan(ious)] = 1
    iou = np.mean(ious)
    return ious[0,0],ious[0,1],iou

def eva(y,pred):
    acc= accuracy_score(y,pred)
    nmi = normalized_mutual_info_score(y, pred)
    ari = adjusted_rand_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    iou_0,iou_1,iou = miou(y,pred)
    return acc,nmi,ari,f1,iou_0,iou_1,iou
