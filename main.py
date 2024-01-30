import torch
from fasterrvit import RViT
from torchvision.models import swin_v2_b, Swin_V2_B_Weights, vit_b_16, ViT_B_16_Weights


swin_model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1, num_classes=1000)
vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT, num_classes=1000)


model = RViT(
    swin_model,
    num_classes=80,
    image_min_size=224,
    image_max_size=224,
    anchor_sizes=(32, 63, 128, 256, 512),
    aspect_ratios=(0.5, 1.0, 2.0),
    rpn_depth=2,
    fpn=True,
    swin_embed_dim=512,
    


)
model.eval()

x = torch.randn(1, 3, 800, 800)
pred = model(x)
print(pred)