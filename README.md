Object detection plays a crucial role in various applications, including autonomous vehicles, surveillance systems, and augmented
reality. Historically, Faster R-CNN demonstrated high precision with
its two-stage architecture, and more recently, Vision Transformers
(ViT) have shown superior performance in specific computer vision
tasks. This study aims to assess the possible benefits of the Region
Proposal Network (RPN) for the precision of Faster R-CNN. Furthermore, this study seeks to leverage the potential advantages of
incorporating transformers into Faster R-CNN, aiming for a highprecision object detection model. The objective is to compare this
model with state-of-the-art counterparts such as YOLOv8 and the
original implementation of Faster R-CNN using ResNet-50. Also, a
comprehensive error analysis will be conducted. This leads to the
formulation of this study’s research question, “How does the performance of R-ViT, which incorporates transformer models in Faster R-CNN,
compare to state-of-the-art object detection?”. The findings reveal that
using a Swin Transformer (Shifted Windows Transformer) using a
Feature Pyramid Network (FPN) as a backbone improves the model
compared to the ResNet-50 variant, particularly excelling in handling
small objects and proving most effective for images with numerous
objects. However, compared to the current state-of-the-art models, it
falls behind on mean Average Precision (mAP). The best-performing
Faster R-CNN model that incorporates transformer-based feature
extractors is the Faster R-CNN (Swin-FPN) model, achieving a mAP
of 0.47.