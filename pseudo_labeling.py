import torch
import numpy as np
from torchvision.ops import box_iou, nms

def calculate_iou(box1, box2):
    """
    Computes IoU between two boxes in [xmin, ymin, xmax, ymax] format.
    """
    b1 = torch.tensor(box1).view(-1, 4)
    b2 = torch.tensor(box2).view(-1, 4)
    return box_iou(b1, b2).numpy()

def generate_pseudo_labels(model, dataloader, device, threshold_strategy="fixed", 
                            fixed_val=0.85, adaptive_percentile=70, epoch=1, total_epochs=10):
    """
    Runs inference on unlabeled images and generates pseudo-labels.
    Supports 'fixed' and 'adaptive' thresholding.
    
    Returns:
        dict: A dictionary mapping image_id to a list of bounding boxes [xmin, ymin, xmax, ymax] and confidence scores.
    """
    model.eval()
    temp_results = {}
    all_scores = []
    
    # Step 1: Collect predictions and all confidence scores
    with torch.no_grad():
        for images, _, img_ids in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for idx, output in enumerate(outputs):
                img_id = img_ids[idx]
                boxes = output['boxes'].cpu()
                labels = output['labels'].cpu()
                scores = output['scores'].cpu()
                
                # Keep only WBC class (class 1)
                wbc_mask = (labels == 1) & (scores > 0.1) # Soft pre-filtering
                boxes = boxes[wbc_mask]
                scores = scores[wbc_mask]
                
                # Apply NMS to remove duplicates before thresholding
                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep].numpy()
                scores = scores[keep].numpy()
                
                temp_results[img_id] = {
                    "boxes": boxes,
                    "scores": scores
                }
                
                all_scores.extend(scores.tolist())

    # Step 2: Determine confidence threshold
    if threshold_strategy == "fixed":
        threshold = fixed_val
    elif threshold_strategy == "adaptive":
        # Adaptive Thresholding:
        # If we don't have predictions, default to fixed_val.
        # Otherwise, use statistical adaptive threshold: mean - 1.0 * std (bounded between 0.50 and 0.90)
        # Or percentile-based. Let's implement a mix of both:
        if len(all_scores) > 0:
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores) if len(all_scores) > 1 else 0
            
            # Epoch-based warming up threshold: starting lower to accumulate pseudo-labels,
            # then becoming more selective in later epochs.
            warmup_factor = min(1.0, epoch / max(1, total_epochs * 0.6))
            adaptive_val = mean_score - 0.5 * std_score
            
            # Bound the threshold between 0.50 and 0.88 to ensure feasibility
            threshold = float(np.clip(adaptive_val * warmup_factor, 0.50, 0.88))
        else:
            threshold = fixed_val
    else:
        raise ValueError(f"Unknown strategy: {threshold_strategy}")
        
    # Step 3: Filter predictions using the determined threshold
    final_pseudo_labels = {}
    for img_id, pred in temp_results.items():
        boxes = pred["boxes"]
        scores = pred["scores"]
        
        keep_mask = scores >= threshold
        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]
        
        final_pseudo_labels[img_id] = {
            "boxes": filtered_boxes,
            "scores": filtered_scores
        }
        
    return final_pseudo_labels, threshold


def evaluate_pseudo_label_quality(pseudo_labels, dataset, iou_threshold=0.5):
    """
    Evaluates the precision, recall, and F1-score of the generated pseudo-labels
    against the hidden ground truth of the unlabeled pool.
    
    pseudo_labels: dict of image_id -> {"boxes": np.array, "scores": np.array}
    dataset: WBCDataset object containing hidden ground truth
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Create dictionary from dataset for quick ground truth lookup
    gt_dict = {}
    for idx in range(len(dataset)):
        _, target, img_id = dataset[idx]
        gt_dict[img_id] = target["boxes"].numpy()

    for img_id, pl in pseudo_labels.items():
        pl_boxes = pl["boxes"]  # Array of [xmin, ymin, xmax, ymax]
        gt_boxes = gt_dict.get(img_id, np.zeros((0, 4)))
        
        num_pl = len(pl_boxes)
        num_gt = len(gt_boxes)
        
        if num_pl == 0:
            total_fn += num_gt
            continue
        if num_gt == 0:
            total_fp += num_pl
            continue
            
        # Compute pairwise IoU
        ious = box_iou(torch.tensor(pl_boxes), torch.tensor(gt_boxes)).numpy()
        
        # Greedy matching
        matched_gt = set()
        tp = 0
        for pl_idx in range(num_pl):
            best_iou = -1
            best_gt_idx = -1
            for gt_idx in range(num_gt):
                if gt_idx in matched_gt:
                    continue
                iou = ious[pl_idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                
        fp = num_pl - tp
        fn = num_gt - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }
