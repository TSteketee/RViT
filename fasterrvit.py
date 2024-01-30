from typing import Dict, List, Optional, Tuple
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch import Tensor, nn

from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from torchvision.models import VisionTransformer, SwinTransformer

from torch.cuda import device
import warnings

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
# import matplotlib.pyplot as plt
from torchvision.ops.misc import Permute
# from torchvision.models.detection.faster_rcnn import RPNHead
from torch import nn

# Imports for Eval Mode
from typing import Tuple, List, Dict
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

# NEW CONV HEAD
from transformers import BigBirdPegasusConfig
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import BigBirdPegasusEncoder




# [] Make it compatible with SwinTransformer
# [] Make it compatible with VisionTransformer
# [] Make it compatible with Swin-FPN
# [] Make it compatible with Vision-FPN
# [] Make focal loss optional
# [] Make the num_classes optional
# [] Make the BigBird RPN optional


# BigBird
class BigBirdEnc(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BigBirdPegasusEncoder(config=config)
        # del self.encoder.embed_tokens

        torch.cuda.empty_cache()

    def forward(self, x):
        x = self.encoder(inputs_embeds = x)
        return x.last_hidden_state

class RPNHeadNew(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, enc_depth=2, hidden_dim = 1024) -> None:
        super().__init__()
        # convs = []
        # for _ in range(conv_depth):
        #     convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        # self.conv = nn.Sequential(*convs)


        self.conv = BigBirdEnc(
            config= BigBirdPegasusConfig(
            hidden_size=hidden_dim,
            encoder_layers=enc_depth,
            encoder_attention_heads=2,
            max_position_embeddings=21824,
            encoder_ffn_dim=hidden_dim,
        ))


        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reshape_input(self, x: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        feature_list = []
        original_sizes = []
        for feature in x:
            orig_size = feature.size()
            original_sizes.append(orig_size[-1])
            new_size = feature.size()[-1]**2
            feature = feature.view(orig_size[0], orig_size[1], new_size)
            feature = feature.permute(0,2,1)
            feature_list.append(feature)
        feature_list = torch.cat(feature_list, dim=1)
        return feature_list, original_sizes
    
    def inverse_reshape_input(self, x: Tensor, oz: List[Tensor]) -> Tensor:
        oz = [z**2 for z in oz]
        assert sum(oz) == x.size(1), "Invalid chunk sizes"
        x = torch.split(x, oz, dim=1)      
        z = []
        for y in x:
            y = y.permute(0, 2, 1)
            orig_size = y.size()        
            new_size_sqrt = int(orig_size[-1] ** 0.5)        
            y = y.view(orig_size[0], orig_size[1], new_size_sqrt, new_size_sqrt)
            z.append(y)
        return z

    # def cat_feature_maps(self, x:List[Tensor]) -> Tensor:



    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        feature, oz = self.reshape_input(x)

        t = self.conv(feature)
        t = self.inverse_reshape_input(feature, oz)
        for x in t:
            logits.append(self.cls_logits(x))
            bbox_reg.append(self.bbox_pred(x))
        return logits, bbox_reg

class BigBirdBoxHead(nn.Module):
    def __init__(self, hidden_size_enc, nhead = 2, num_layers = 2, hidden_dim = 1024):
        super(BigBirdBoxHead, self).__init__()
        
        self.flatten1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.permute = Permute([0, 2, 1])
        self.transformer_encoder = BigBirdEnc(
            config= BigBirdPegasusConfig(
            hidden_size=hidden_dim,
            encoder_layers=nhead,
            encoder_attention_heads=num_layers,
            max_position_embeddings=1000,
            encoder_ffn_dim=hidden_dim,
        ))
        self.flatten2 = nn.Flatten()
        self.linear = nn.Linear(hidden_size_enc * (7**2), 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten1(x)
        x = self.permute(x)
        x = self.transformer_encoder(x)
        x = self.flatten2(x)
        x = self.linear(x)
        x = self.relu(x)
        return x

class SwinFPN(torch.nn.Module):
    def __init__(self, swin_model, hidden_dim = 1024):
        super().__init__()
        swin_model = swin_model.features
        self.out_channels = hidden_dim

        self.swin_model = IntermediateLayerGetter(swin_model, return_layers={
            '1': '0',
            '3': '1', 
            '5': '2', 
            '7': '3'}
            )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
            norm_layer=torch.nn.BatchNorm2d
        )

        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

    def forward(self, x):
        x = self.swin_model(x)
        # show me the shapes of the values in x(orderd dict)

        # x is a ordered dict of 4 layers, permute every layer using self.permute and return it as a ordered dict
        x = OrderedDict([(k, self.permute(v)) for k, v in x.items()])

        x = self.fpn(x)
        return x

class CustomVit(nn.Module):
    def __init__(self, vit_model):
        super(CustomVit, self).__init__()
        # remove the classification head
        self.transformer = vit_model

        def custom_ViT_forward(self, x: torch.Tensor):
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)

            # Classifier "token" as used by standard language architectures
            # x = x[:, 0]

            # instead of isolating the classifier token, we remove the class token from the output
            x = x[:, 1:]


            # x = self.heads(x)

            return x
        
        self.apply_custom_forward(vit_model, custom_ViT_forward)
    
    def apply_custom_forward(self, vit_model, custom_ViT_forward):
        vit_model.forward = custom_ViT_forward.__get__(vit_model)

    def reshape_to_2d(self, x):
        n, p, h = x.shape
        p_s = torch.sqrt(torch.tensor(p)).int()
        x = x.permute(0, 2, 1)
        x = x.view(n, h, p_s, p_s)
        return x

    def forward(self, x):
        x = self.transformer(x)
        x = self.reshape_to_2d(x)
        return x

class CustomSwin(nn.Module):
    # we should take swin model and remove the head. Then recontruct into x by x square just like custom vit
    def __init__(self, swin_model):
        super(CustomSwin, self).__init__()
        # remove the classification head
        self.transformer = swin_model
        self.transformer.permute = torch.nn.Identity()
        self.transformer.avgpool = torch.nn.Identity()
        self.transformer.norm = torch.nn.Identity()
        self.transformer.flatten = torch.nn.Identity()
        self.transformer.head = torch.nn.Identity()


    def reshape_to_2d(self, x):
        n, p_w, p_h, h = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(n, h, p_w, p_h)
        return x
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.reshape_to_2d(x)
        return x


class RViT(torchvision.models.detection.FasterRCNN):
    def __init__(self, 
                 vit_model, 
                 num_classes, 
                 roi_output_size: Optional[int] = 7,
                 image_min_size: Optional[int] = 224, 
                 image_max_size: Optional[int] = 224, 
                 anchor_sizes: tuple = (2, 4, 6, 7), 
                 aspect_ratios: tuple = (0.5, 1.0, 2.0), 
                 rpn_depth: Optional[int] = 1,
                 vit_trainable_layers: Optional[int] = 0,
                 rpn_pre_nms_top_n_train: Optional[int] = 2000,
                 rpn_pre_nms_top_n_test: Optional[int] = 1000,
                 rpn_post_nms_top_n_train: Optional[int] = 2000,
                 rpn_post_nms_top_n_test: Optional[int] = 1000,
                 fpn: bool = False, # For now only possible with SwinTransformer
                 swin_embed_dim: Optional[int] = 512,
                 skip_resize: bool = False, # If true, the image is not resized to min_size and max_size within the GeneralizedRCNNTransform
                 size_divisible: Optional[int] = 32, # If the size of the image is not divisible by this number, the image is padded with zeros for onnx
                 transformer_rpn: bool = False, # If true, the RPN head is replaced with a transformer based RPN head
                 transformer_det_head = False, # If true, the detection head is replaced with a transformer based detection head
                 transformer_det_head_num_layers = 2, # Number of layers in the transformer based detection head
                 transformer_det_head_nhead = 2, # Number of heads in the transformer based detection head

                 ):
        
        
        

        if isinstance(vit_model, VisionTransformer):
            if fpn:
                print("FPN is not yet available with plain Vision Transformers.")
                raise NotImplementedError
            else:
                print("Initializing RVit withf a plain Vision Transformer model...")
                backbone = CustomVit(vit_model)
                backbone.out_channels = vit_model.hidden_dim

        elif isinstance(vit_model, SwinTransformer):
            if fpn:
                print("Initializing RVit with a Swin-FPN Transformer...")
                backbone = SwinFPN(vit_model, hidden_dim=swin_embed_dim)
                backbone.out_channels = swin_embed_dim
            else:  
                print("Initializing RVit with a plain Swin Transformer...")
                out_channels = vit_model.head.in_features
                backbone = CustomSwin(vit_model)
                backbone.out_channels = out_channels
        else:
            raise ValueError("vit_model must be either a VisionTransformer or a SwinTransformer")


        if isinstance(vit_model, VisionTransformer):
            pass
        elif isinstance(vit_model, SwinTransformer):
            # turn anchor_sizes into a tuple of tuples of integers
            if fpn:
                anchor_sizes = (anchor_sizes,) * 5
                aspect_ratios = (aspect_ratios,) * len(anchor_sizes)
                # anchor_sizes = tuple((x,) for x in anchor_sizes)
                # anchor_sizes = anchor_sizes * 5

                # aspect_ratios = tuple((x,) for x in aspect_ratios)
                # aspect_ratios = aspect_ratios * 5

                # print(anchor_sizes)
                # print(aspect_ratios)


        if fpn:
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        else:
            anchor_generator = AnchorGenerator(sizes=(anchor_sizes,), aspect_ratios=(aspect_ratios,))

        if transformer_rpn:
            rpn_head= RPNHeadNew(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], enc_depth=rpn_depth
            )
        else:
            rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=rpn_depth)


        # ROI Heads
        
        
        if transformer_det_head: 
            roi_heads = BigBirdBoxHead(
                hidden_size_enc=backbone.out_channels,
                num_layers=transformer_det_head_num_layers,
                nhead=transformer_det_head_nhead,
            )       

        if fpn:
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=roi_output_size, sampling_ratio=2)
        else:
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=roi_output_size, sampling_ratio=2)

        self.skip_resize = skip_resize
        self.size_divisible = size_divisible

        super(RViT, self).__init__(
            # backbone
            backbone = backbone,

            # rpn
            rpn_head= rpn_head,
            rpn_anchor_generator=anchor_generator,

            # roi_heads
            box_roi_pool=roi_pooler,
            box_head=roi_heads if transformer_det_head else TwoMLPHead(backbone.out_channels * roi_output_size ** 2, 1024),
            num_classes=num_classes,
            min_size=image_min_size,
            max_size=image_max_size,
            rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train = rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test = rpn_post_nms_top_n_test,

            _skip_resize = self.skip_resize,
            size_divisible = size_divisible,
        )


        # Freeze the layers
        self.freeze_layers(vit_model, trainable_layers=vit_trainable_layers)

    def freeze_layers(self, vit_model, trainable_layers):
        #VisionTransformer
        if isinstance(vit_model, torchvision.models.vision_transformer.VisionTransformer):
            if trainable_layers > 0:
                for layer in vit_model.encoder.layers[:-trainable_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for param in vit_model.parameters():
                    param.requires_grad = False
                    
        #SwinTransformer
        elif isinstance(vit_model, torchvision.models.swin_transformer.SwinTransformer):
            if trainable_layers > 0:
                for layer in vit_model.layers[:-trainable_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for param in vit_model.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("vit_model must be either a VisionTransformer or a SwinTransformer")
        

    

    def evaluation_losses_forward(self, images, targets):

        original_image_sizes: List[Tuple[int, int]] = []

        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)




        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        # self.rpn.training=True
        #self.roi_heads.training=True


        #####proposals, proposal_losses = self.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = self.rpn.head(features_rpn)
        anchors = self.rpn.anchor_generator(images, features_rpn)


        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = self.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = self.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = self.roi_heads.select_training_samples(proposals, targets)
        box_features = self.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        self.rpn.training=False
        self.roi_heads.training=False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections