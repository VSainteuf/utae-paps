"""
PaPs Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from src.backbones.utae import ConvLayer


class PaPs(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes=20,
        shape_size=16,
        mask_conv=True,
        min_confidence=0.2,
        min_remain=0.5,
        mask_threshold=0.4,
    ):
        """
        Implementation of the Parcel-as-Points Module (PaPs) for panoptic segmentation of agricultural
        parcels from satellite image time series.
        Args:
            encoder (nn.Module): Backbone encoding network. The encoder is expected to return
            a feature map at the same resolution as the input images and a list of feature maps
            of lower resolution.
            num_classes (int): Number of classes (including stuff and void classes).
            shape_size (int): S hyperparameter defining the shape of the local patch.
            mask_conv (bool): If False no residual CNN is applied after combination of
            the predicted shape and the cropped saliency (default True)
            min_confidence (float): Cut-off confidence level for the pseudo NMS (predicted instances with
            lower condidence will not be included in the panoptic prediction).
            min_remain (float): Hyperparameter of the pseudo-NMS that defines the fraction of a candidate instance mask
            that needs to be new to be included in the final panoptic prediction (default  0.5).
            mask_threshold (float): Binary threshold for instance masks (default 0.4)

        """
        super(PaPs, self).__init__()
        self.encoder = encoder
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.min_scale = 1 / shape_size
        self.register_buffer("min_confidence", torch.tensor([min_confidence]))
        self.min_remain = min_remain
        self.mask_threshold = mask_threshold
        self.center_extractor = CenterExtractor()

        enc_dim = encoder.enc_dim
        stack_dim = encoder.stack_dim
        self.heatmap_conv = nn.Sequential(
            ConvLayer(nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1,
                      padding_mode="reflect"),
            nn.Sigmoid(),
        )

        self.saliency_conv = ConvLayer(
            nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1,
            padding_mode="reflect"
        )

        self.shape_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, shape_size ** 2),
        )

        self.size_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.BatchNorm1d(stack_dim // 4),
            nn.ReLU(),
            nn.Linear(stack_dim // 4, 2),
            nn.Softplus(),
        )

        self.class_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.Linear(stack_dim // 4, num_classes),
        )

        if mask_conv:
            self.mask_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=16, num_groups=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            )
        else:
            self.mask_cnn = None

    def forward(
        self,
        input,
        batch_positions=None,
        zones=None,
        pseudo_nms=True,
        heatmap_only=False,
    ):
        """
        Args:
            input (tensor): Input image time series.
            batch_positions (tensor): Date sequence of the batch images.
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection). This mapping
            is used at train time to predict and supervise at most one prediction
            per ground truth object for efficiency.
            When not provided all predicted centers receive supervision.
            pseudo_nms (bool): If True performs pseudo_nms to produce a panoptic prediction,
            otherwise the model returns potentially overlapping instance segmentation masks (default True).
            heatmap_only (bool): If True the model only returns the centerness heatmap. Can be useful for some
            warmup epochs of the centerness prediction, as all the rest hinges on this.

        Returns:
            predictions (dict[tensor]): A dictionary of predictions with the following keys:
                center_mask         (B,H,W) Binary mask of centers.
                saliency            (B,1,H,W) Global Saliency.
                heatmap             (B,1,H,W) Predicted centerness heatmap.
                semantic            (M, K) Predicted class scores for each center (with M the number of predicted centers).
                size                (M, 2) Predicted sizes for each center.
                confidence          (M,1) Predicted centerness for each center.
                centerness          (M,1) Predicted centerness for each center.
                instance_masks      List of N binary masks of varying shape.
                instance_boxes      (N, 4) Coordinates of the N bounding boxes.
                pano_instance       (B,H,W) Predicted instance id for each pixel.
                pano_semantic       (B,K,H,W) Predicted class score for each pixel.

        """
        out, maps = self.encoder(input, batch_positions=batch_positions)

        # Global Predictions
        heatmap = self.heatmap_conv(out)
        saliency = self.saliency_conv(out)

        center_mask, _ = self.center_extractor(
            heatmap, zones=zones
        )  # (B,H,W) mask of N detected centers
        center_mask = center_mask.squeeze()

        if heatmap_only:
            predictions = dict(
                center_mask=center_mask,
                saliency=None,
                heatmap=heatmap,
                semantic=None,
                size=None,
                offsets=None,
                confidence=None,
                instance_masks=None,
                instance_boxes=None,
                pano_instance=None,
                pano_semantic=None,
            )
            return predictions

        # Retrieve info of detected centers
        H, W = heatmap.shape[-2:]
        center_batch, center_h, center_w = torch.where(center_mask)
        center_positions = torch.stack([center_h, center_w], dim=1)

        # Construct multi-level feature stack for centers
        stack = []
        for i, m in enumerate(maps):
            h_mask = center_h // (2 ** (len(maps) - 1 - i))
            # Assumes resolution is divided by 2 at each level
            w_mask = center_w // (2 ** (len(maps) - 1 - i))
            m = m.permute(0, 2, 3, 1)
            stack.append(m[center_batch, h_mask, w_mask])
        stack = torch.cat(stack, dim=1)

        # Center-level predictions
        size = self.size_mlp(stack)
        sem = self.class_mlp(stack)
        shapes = self.shape_mlp(stack)
        shapes = shapes.view((-1, 1, self.shape_size, self.shape_size))
        # (N,1,S,S) instance shapes

        centerness = heatmap[center_mask[:, None, :, :]].unsqueeze(-1)
        confidence = centerness

        # Instance Boxes Assembling
        ## Minimal box size of 1px
        ## Combine clamped sizes and center positions to obtain box coordinates
        clamp_size = size.detach().round().long().clamp_min(min=1)
        half_size = clamp_size // 2
        remainder_size = clamp_size % 2
        start_hw = center_positions - half_size
        stop_hw = center_positions + half_size + remainder_size

        instance_boxes = torch.cat([start_hw, stop_hw], dim=1)
        instance_boxes.clamp_(min=0, max=H)
        instance_boxes = instance_boxes[:, [1, 0, 3, 2]]  # h,w,h,w to x,y,x,y

        valid_start = (-start_hw).clamp_(
            min=0
        )  # if h=-5 crop the shape mask before the 5th pixel
        valid_stop = (stop_hw - start_hw) - (stop_hw - H).clamp_(
            min=0
        )  # crop if h_stop > H

        # Instance Masks Assembling
        instance_masks = []
        for i, s in enumerate(shapes.split(1, dim=0)):
            h, w = clamp_size[i]  # Box size
            w_start, h_start, w_stop, h_stop = instance_boxes[i]  # Box coordinates
            h_start_valid, w_start_valid = valid_start[i]  # Part of the Box that lies
            h_stop_valid, w_stop_valid = valid_stop[i]  # within the image's extent

            ## Resample local shape mask
            pred_mask = (
                F.interpolate(s, size=(h.item(), w.item()), mode="bilinear")
            ).squeeze(0)
            pred_mask = pred_mask[
                :, h_start_valid:h_stop_valid, w_start_valid:w_stop_valid
            ]

            ## Crop saliency
            crop_saliency = saliency[center_batch[i], :, h_start:h_stop, w_start:w_stop]

            ## Combine both
            if self.mask_cnn is None:
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(crop_saliency)
            else:
                pred_mask = pred_mask + crop_saliency
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(
                    self.mask_cnn(pred_mask.unsqueeze(0)).squeeze(0)
                )
            instance_masks.append(pred_mask)

        # PSEUDO-NMS
        if pseudo_nms:
            panoptic_instance = []
            panoptic_semantic = []
            for b in range(center_mask.shape[0]):  # iterate over elements of batch
                panoptic_mask = torch.zeros(
                    center_mask[0].shape, device=center_mask.device
                )
                semantic_mask = torch.zeros(
                    (self.num_classes, *center_mask[0].shape), device=center_mask.device
                )

                candidates = torch.where(center_batch == b)[
                    0
                ]  # get indices of centers in this batch element
                for n, (c, idx) in enumerate(
                    zip(*torch.sort(confidence[candidates].squeeze(), descending=True))
                ):
                    if c < self.min_confidence:
                        break
                    else:
                        new_mask = torch.zeros(
                            center_mask[0].shape, device=center_mask.device
                        )
                        pred_mask_bin = (
                            instance_masks[candidates[idx]].squeeze(0)
                            > self.mask_threshold
                        ).float()

                        if pred_mask_bin.sum() > 0:
                            xtl, ytl, xbr, ybr = instance_boxes[candidates[idx]]
                            new_mask[ytl:ybr, xtl:xbr] = pred_mask_bin

                            if ((new_mask != 0) * (panoptic_mask != 0)).any():
                                n_total = (new_mask != 0).sum()
                                non_overlaping_mask = (new_mask != 0) * (
                                    panoptic_mask == 0
                                )
                                n_new = non_overlaping_mask.sum().float()
                                if n_new / n_total > self.min_remain:
                                    panoptic_mask[non_overlaping_mask] = n + 1
                                    semantic_mask[:, non_overlaping_mask] = sem[
                                        candidates[idx]
                                    ][:, None]
                            else:
                                panoptic_mask[(new_mask != 0)] = n + 1
                                semantic_mask[:, (new_mask != 0)] = sem[
                                    candidates[idx]
                                ][:, None]
                panoptic_instance.append(panoptic_mask)
                panoptic_semantic.append(semantic_mask)
            panoptic_instance = torch.stack(panoptic_instance, dim=0)
            panoptic_semantic = torch.stack(panoptic_semantic, dim=0)
        else:
            panoptic_instance = None
            panoptic_semantic = None

        predictions = dict(
            center_mask=center_mask,
            saliency=saliency,
            heatmap=heatmap,
            semantic=sem,
            size=size,
            confidence=confidence,
            centerness=centerness,
            instance_masks=instance_masks,
            instance_boxes=instance_boxes,
            pano_instance=panoptic_instance,
            pano_semantic=panoptic_semantic,
        )

        return predictions


class CenterExtractor(nn.Module):
    def __init__(self):
        """
        Module for local maxima extraction
        """
        super(CenterExtractor, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, input, zones=None):
        """

        Args:
            input (tensor): Centerness heatmap
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection).
            If provided, the highest local maxima in each zone is kept. As a result at most one
            prediction is made per ground truth object.
            If not provided, all local maxima are returned.
        """
        if zones is not None:
            masks = []
            for b, x in enumerate(input.split(1, dim=0)):
                x = x.view(-1)
                _, idxs = scatter_max(x, zones[b].view(-1).long())
                mask = torch.zeros(x.shape, device=x.device)
                mask[idxs[idxs != x.shape[0]]] = 1
                masks.append(mask.view(zones[b].shape).unsqueeze(0))
            centermask = torch.stack(masks, dim=0).bool()
        else:
            centermask = input == self.pool(input)
            no_valley = input > input.mean()
            centermask = centermask * no_valley

        n_centers = int(centermask.sum().detach().cpu())
        return centermask, n_centers
