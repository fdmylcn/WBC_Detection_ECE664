import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(backbone_type="resnet50", num_classes=2, pretrained=True):
    """
    Constructs a Faster R-CNN model for WBC detection.
    Supports a heavy, robust ResNet50 backbone or a lightweight, fast MobileNetV3 backbone.
    
    num_classes = 2 (class 0 is background, class 1 is WBC)
    """
    if backbone_type == "resnet50":
        # Load standard pre-trained Faster R-CNN model
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        else:
            weights = None
            
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace the classifier head with a new one for single-class WBC (2 classes total)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif backbone_type == "mobilenet_v3":
        # Lightweight backbone for ultra-fast training/CPU execution
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        else:
            weights = None
            
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        
        # Replace classification and box regression head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")
        
    return model
