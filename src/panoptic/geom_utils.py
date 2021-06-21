import torch
import numpy as np


def get_bbox(bin_mask):
    """Input single (H,W) bin mask"""
    if isinstance(bin_mask, torch.Tensor):
        xl, xr = torch.where(bin_mask.sum(dim=-2))[0][[0, -1]]
        yt, yb = torch.where(bin_mask.sum(dim=-1))[0][[0, -1]]
        return torch.stack([xl, yt, xr, yb])
    else:
        xl, xr = np.where(bin_mask.sum(axis=-2))[0][[0, -1]]
        yt, yb = np.where(bin_mask.sum(axis=-1))[0][[0, -1]]
        return np.stack([xl, yt, xr, yb])


def bbox_area(bbox):
    """Input (N,4) set of bounding boxes"""
    out = bbox.float()
    return (out[:, 2] - out[:, 0]) * (out[:, 3] - out[:, 1])


def intersect(box_a, box_b):
    """
     taken from https://github.com/amdegroot/ssd.pytorch
     We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_iou(bbox1, bbox2):
    """Two sets of (N,4) bounding boxes"""
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    inter = torch.diag(intersect(bbox1, bbox2))
    union = area1 + area2 - inter

    return inter[union != 0] / union[union != 0]


def bbox_validzone(bbox, shape):
    """Given an image shape, get the coordinate (in the bbox reference)
    of the pixels that are within the image boundaries"""
    H, W = shape
    wt, ht, wb, hb = bbox

    val_ht = -ht if ht < 0 else 0
    val_wt = -wt if wt < 0 else 0
    val_hb = H - ht if hb > H else hb - ht
    val_wb = W - wt if wb > W else wb - wt
    return (val_wt, val_ht, val_wb, val_hb)
