import torch
import torch.nn as nn
from torchvision import transforms
from metrics import AlignmentMetrics
from calibrated_similarity import calibrate

def cka(feat_A, feat_B):
    eval_fn = lambda x,y: AlignmentMetrics.measure("cka", x, y, kernel_metric='ip')
    return calibrate(feat_A, feat_B, eval_fn)

def mknn(feat_A, feat_B, k=10):
    eval_fn = lambda x,y: AlignmentMetrics.measure("mutual_knn", x, y, topk=k)
    return calibrate(feat_A, feat_B, eval_fn)
