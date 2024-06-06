import torch
from utils.metrics import bbox_iou

pbox = torch.tensor((1, 2, 3, 4))
tbox = torch.tensor((2, 3, 1, 4))

iou = bbox_iou(pbox, tbox, DIoU=True)
from utils.segment.loss import ComputeLoss