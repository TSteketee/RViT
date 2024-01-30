# System imports
import os
import sys

# Models imports
import torch
from torchvision.models.swin_transformer import swin_v2_b, Swin_V2_B_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.rpn import AnchorGenerator
import matplotlib.pyplot as plt
from torchvision.ops.misc import Permute
from torchvision.models.detection.faster_rcnn import RPNHead

# Imports for Eval Mode
from typing import Tuple, List, Dict
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

# Loader imports
from tqdm import tqdm

# Metrics imports
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Loss imports
import torchvision
# from ciou_loss import complete_box_iou_loss
from torch.nn import functional as F
from focal_loss import sigmoid_focal_loss, softmax_focal_loss

torch.autograd.set_detect_anomaly(True)

# EMPTY CUDA
torch.cuda.empty_cache()


# HYPERPARAMS
IMG_SIZE = 800
NUM_EPOCHS = 50
BATCH_SIZE = 32


# MODEL
swin_model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT, num_classes=1000)
swin_model = swin_model.features




backbone_swin = SwinFPN(swin_model)

# MONKEY PATCHING LOSS FRCNN
def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # orig_classification_loss = F.cross_entropy(class_logits, labels)

    classification_loss = softmax_focal_loss(class_logits, labels)

    # print("----------------------------------------------------------")
    # print("origional classification loss", orig_classification_loss)
    # print("custom classification loss", classification_loss)
    # print("----------------------------------------------------------")


    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape     
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    # check if the box loss is nan
    if torch.isnan(box_loss):
        print("DETHEAD: box loss is nan")
        print(box_loss)

    if torch.isnan(classification_loss):
        print("DETHEAD: classification loss is nan")
        print(classification_loss)


    return classification_loss, box_loss


def custom_rpn_loss(
    self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        objectness (Tensor)
        pred_bbox_deltas (Tensor)
        labels (List[Tensor])
        regression_targets (List[Tensor])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor)
    """

    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)



    box_loss = F.smooth_l1_loss(
        pred_bbox_deltas[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1 / 9,
        reduction="sum",
    ) / (sampled_inds.numel())


    objectness_loss = sigmoid_focal_loss(objectness[sampled_inds], labels[sampled_inds])
    objectness_loss *= 7
    # orig_objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
    
    if torch.isnan(box_loss):
        print("RPN: box loss is nan")
        print(box_loss)

    if torch.isnan(objectness_loss):
        print("RPN: objectness loss is nan")
        print(objectness_loss)


    return objectness_loss, box_loss



torchvision.models.detection.roi_heads.fastrcnn_loss = custom_fastrcnn_loss


model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, num_classes=91)

# change the model to have 25 classes instead of 91
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=1024, out_features=25, bias=True)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features=1024, out_features=100, bias=True)

model.backbone = backbone_swin

# freeze backbone
for param in model.backbone.swin_model.parameters():
    param.requires_grad = False

# adjust rpn anchor sizes 
# anchor_sizes = ((2, 5, 10, 20, 40),) * 5
anchor_sizes = ((4, 6, 12, 20, 40),) * 5

aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)


# adjust rpn head accorcing to the new anchor sizes
model.rpn.head = RPNHead(
    model.backbone.out_channels,
    model.rpn.anchor_generator.num_anchors_per_location()[0], 
    conv_depth=2
)

# ADD CUSTOM RPN LOSS
model.rpn.compute_loss = custom_rpn_loss.__get__(model.rpn)

# model.load_state_dict(torch.load('rvit_model_swin_b_fpn.pth'))

# Calculate FLOPS
from calflops import calculate_flops
flops = calculate_flops(model, (1, 3, IMG_SIZE, IMG_SIZE))
print(flops)

exit()