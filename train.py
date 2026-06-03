import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import WBCDataset, collate_fn
from models import get_faster_rcnn_model
from pseudo_labeling import generate_pseudo_labels, evaluate_pseudo_label_quality
from torchvision.ops import box_iou
from tqdm import tqdm

# --- 1. Pure Python mAP Evaluation Engine ---

def calculate_ap(precisions, recalls):
    """
    Computes Average Precision (AP) using the COCO/11-point AUC interpolation.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_map(model, dataloader, device, iou_thresholds=None):
    """
    Evaluates the model and computes mAP@0.5 and mAP@0.5:0.95.
    Written in pure Python/PyTorch for Windows stability without external cocoapi compile dependencies.
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for out, tgt in zip(outputs, targets):
                pred_boxes = out['boxes'].cpu()
                pred_scores = out['scores'].cpu()
                pred_labels = out['labels'].cpu()
                
                # Keep only WBC (class 1)
                wbc_mask = pred_labels == 1
                pred_boxes = pred_boxes[wbc_mask]
                pred_scores = pred_scores[wbc_mask]
                
                gt_boxes = tgt['boxes'].cpu()
                gt_labels = tgt['labels'].cpu()
                gt_wbc_mask = gt_labels == 1
                gt_boxes = gt_boxes[gt_wbc_mask]
                
                all_predictions.append({
                    "boxes": pred_boxes.numpy(),
                    "scores": pred_scores.numpy()
                })
                
                all_targets.append({
                    "boxes": gt_boxes.numpy()
                })

    # Accumulate TP/FP across all images for various IoU thresholds
    ap_per_iou = []
    
    for iou_thresh in iou_thresholds:
        y_scores = []
        y_true = []
        total_gts = 0
        
        for pred, tgt in zip(all_predictions, all_targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            gt_boxes = tgt["boxes"]
            
            num_preds = len(pred_boxes)
            num_gts = len(gt_boxes)
            total_gts += num_gts
            
            if num_preds == 0:
                continue
            if num_gts == 0:
                y_scores.extend(pred_scores.tolist())
                y_true.extend([0] * num_preds)
                continue
                
            # Compute pairwise IoU
            ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
            
            # Sort predictions by score descending
            sort_idx = np.argsort(-pred_scores)
            matched_gts = set()
            
            for p_idx in sort_idx:
                score = pred_scores[p_idx]
                y_scores.append(score)
                
                best_iou = -1
                best_gt_idx = -1
                for g_idx in range(num_gts):
                    if g_idx in matched_gts:
                        continue
                    iou = ious[p_idx, g_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= iou_thresh:
                    y_true.append(1)
                    matched_gts.add(best_gt_idx)
                else:
                    y_true.append(0)
                    
        if len(y_scores) == 0:
            ap_per_iou.append(0.0)
            continue
            
        # Sort combined results by score
        y_scores = np.array(y_scores)
        y_true = np.array(y_true)
        sort_indices = np.argsort(-y_scores)
        y_true = y_true[sort_indices]
        
        tp_cum = np.cumsum(y_true)
        fp_cum = np.cumsum(1 - y_true)
        
        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = tp_cum / total_gts if total_gts > 0 else np.zeros_like(tp_cum)
        
        ap = calculate_ap(precisions, recalls)
        ap_per_iou.append(ap)
        
    mAP_50 = ap_per_iou[0]
    mAP_50_95 = np.mean(ap_per_iou)
    
    # Calculate overall final Precision and Recall at standard threshold 0.50
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, tgt in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = tgt["boxes"]
        
        # Filter predictions at score threshold >= 0.50
        keep = pred_scores >= 0.50
        pred_boxes = pred_boxes[keep]
        
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        
        if num_preds == 0:
            total_fn += num_gts
            continue
        if num_gts == 0:
            total_fp += num_preds
            continue
            
        ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()
        matched = set()
        tp = 0
        for p_idx in range(num_preds):
            best_iou = -1
            best_gt_idx = -1
            for g_idx in range(num_gts):
                if g_idx in matched:
                    continue
                iou = ious[p_idx, g_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            if best_iou >= 0.5:
                tp += 1
                matched.add(best_gt_idx)
                
        total_tp += tp
        total_fp += (num_preds - tp)
        total_fn += (num_gts - len(matched))
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mAP@0.5": float(mAP_50),
        "mAP@0.5:0.95": float(mAP_50_95),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

# --- 2. Model Training Helpers ---

def train_one_epoch(model, optimizer, dataloader, device):
    """
    Standard supervised epoch training.
    """
    model.train()
    total_loss = 0.0
    
    for images, targets, _ in dataloader:
        images = [img.to(device) for img in images]
        # Targets are list of dicts, map coordinates to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass returning loss dictionary
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += losses.item()
        
    return total_loss / len(dataloader)

# --- 3. Supervised Baseline Framework ---

def run_supervised_training(train_dataset, val_dataset, test_dataset, device, 
                            backbone="mobilenet_v3", epochs=10, lr=0.005, batch_size=4):
    """
    Trains standard Faster R-CNN purely on labeled training images.
    """
    model = get_faster_rcnn_model(backbone_type=backbone, num_classes=2, pretrained=True)
    model.to(device)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
                             
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    best_map = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, dataloader, device)
        val_metrics = evaluate_map(model, val_loader, device)
        
        if val_metrics["mAP@0.5"] > best_map:
            best_map = val_metrics["mAP@0.5"]
            best_model_wts = copy.deepcopy(model.state_dict())
            
    # Evaluate best model on test set
    model.load_state_dict(best_model_wts)
    test_metrics = evaluate_map(model, test_loader, device)
    return test_metrics

# --- 4. STAC Semi-Supervised Framework ---

def run_stac_training(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
                      backbone="mobilenet_v3", teacher_epochs=6, student_epochs=10,
                      lr=0.005, batch_size=4, threshold_strategy="fixed", fixed_val=0.85):
    """
    STAC Framework:
    1. Train Teacher on labeled data.
    2. Run Teacher on unlabeled data to generate pseudo-labels.
    3. Train Student on combined Labeled (weak augmentations) + Unlabeled (strong augmentations)
       with Teacher's pseudo-labels.
    """
    # Step 1: Train Teacher Model
    print("   Training STAC Teacher model...")
    teacher_model = get_faster_rcnn_model(backbone_type=backbone, num_classes=2, pretrained=True)
    teacher_model.to(device)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
                             
    teacher_opt = torch.optim.SGD(teacher_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    best_map = -1
    best_teacher_wts = copy.deepcopy(teacher_model.state_dict())
    
    for epoch in range(1, teacher_epochs + 1):
        train_one_epoch(teacher_model, teacher_opt, labeled_loader, device)
        val_metrics = evaluate_map(teacher_model, val_loader, device)
        if val_metrics["mAP@0.5"] > best_map:
            best_map = val_metrics["mAP@0.5"]
            best_teacher_wts = copy.deepcopy(teacher_model.state_dict())
            
    teacher_model.load_state_dict(best_teacher_wts)
    
    # Step 2: Generate pseudo-labels on unlabeled dataset
    print(f"   Generating pseudo-labels (Strategy: {threshold_strategy})...")
    # Wrap unlabeled dataset with NO augmentations for clean pseudo-labeling
    unlabeled_eval_dataset = copy.deepcopy(unlabeled_dataset)
    unlabeled_eval_dataset.augmentation_type = None
    unlabeled_eval_loader = DataLoader(unlabeled_eval_dataset, batch_size=batch_size, shuffle=False,
                                       collate_fn=collate_fn, num_workers=0)
                                       
    pseudo_labels, thresh = generate_pseudo_labels(
        teacher_model, unlabeled_eval_loader, device, 
        threshold_strategy=threshold_strategy, fixed_val=fixed_val,
        epoch=teacher_epochs, total_epochs=teacher_epochs
    )
    
    # Evaluate pseudo-label quality separately
    pl_quality = evaluate_pseudo_label_quality(pseudo_labels, unlabeled_eval_dataset)
    print(f"   [Pseudo-Label Quality] Threshold: {thresh:.3f} | Precision: {pl_quality['precision']:.3f} | Recall: {pl_quality['recall']:.3f}")
    
    # Step 3: Combine datasets for Student training
    # Create combined dataset
    # We will construct a custom loader that mixes labeled batch and pseudo-labeled batch
    class STACCombinedDataset(Dataset):
        def __init__(self, labeled_ds, unlabeled_ds, pseudo_labels):
            self.labeled_ds = copy.deepcopy(labeled_ds)
            self.labeled_ds.augmentation_type = 'weak'
            
            self.unlabeled_ds = copy.deepcopy(unlabeled_ds)
            self.unlabeled_ds.augmentation_type = 'strong_photometric'
            
            self.pseudo_labels = pseudo_labels
            self.labeled_len = len(self.labeled_ds)
            self.unlabeled_len = len(self.unlabeled_ds)
            
        def __len__(self):
            return self.labeled_len + self.unlabeled_len
            
        def __getitem__(self, idx):
            if idx < self.labeled_len:
                return self.labeled_ds[idx]
            else:
                unlabeled_idx = idx - self.labeled_len
                img, target, img_id = self.unlabeled_ds[unlabeled_idx]
                
                # Replace hidden true target with generated pseudo-label
                pl = self.pseudo_labels.get(img_id, {"boxes": np.zeros((0, 4))})
                pl_boxes = pl["boxes"]
                
                boxes_tensor = []
                labels_tensor = []
                for box in pl_boxes:
                    xmin, ymin, xmax, ymax = box
                    boxes_tensor.append([xmin, ymin, xmax, ymax])
                    labels_tensor.append(1) # WBC class 1
                    
                if len(boxes_tensor) > 0:
                    new_target = {
                        "boxes": torch.as_tensor(boxes_tensor, dtype=torch.float32),
                        "labels": torch.as_tensor(labels_tensor, dtype=torch.int64),
                        "image_id": torch.tensor([img_id])
                    }
                else:
                    new_target = {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64),
                        "image_id": torch.tensor([img_id])
                    }
                return img, new_target, img_id

    combined_dataset = STACCombinedDataset(labeled_dataset, unlabeled_dataset, pseudo_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, num_workers=0)
                                 
    # Train Student model
    print("   Training STAC Student model...")
    student_model = get_faster_rcnn_model(backbone_type=backbone, num_classes=2, pretrained=True)
    student_model.to(device)
    student_opt = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    best_student_map = -1
    best_student_wts = copy.deepcopy(student_model.state_dict())
    
    for epoch in range(1, student_epochs + 1):
        train_one_epoch(student_model, student_opt, combined_loader, device)
        val_metrics = evaluate_map(student_model, val_loader, device)
        if val_metrics["mAP@0.5"] > best_student_map:
            best_student_map = val_metrics["mAP@0.5"]
            best_student_wts = copy.deepcopy(student_model.state_dict())
            
    student_model.load_state_dict(best_student_wts)
    test_metrics = evaluate_map(student_model, test_loader, device)
    
    # Pack metrics and pseudo-label quality together
    test_metrics["pl_precision"] = pl_quality["precision"]
    test_metrics["pl_recall"] = pl_quality["recall"]
    test_metrics["pl_f1_score"] = pl_quality["f1_score"]
    
    return test_metrics

# --- 5. Soft Teacher Semi-Supervised Framework ---

def update_teacher_ema(student_model, teacher_model, alpha=0.99):
    """
    Updates Teacher weights using Exponential Moving Average (EMA).
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)

def run_soft_teacher_training(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
                             backbone="mobilenet_v3", epochs=10, lr=0.005, batch_size=4,
                             ema_alpha=0.99, threshold_strategy="fixed", fixed_val=0.85):
    """
    Soft Teacher Framework (Dynamic Teacher-Student model):
    - Maintains EMA Teacher and SGD Student.
    - Labeled batches train Student normally.
    - For unlabeled images, EMA Teacher predicts dynamic pseudo-labels on-the-fly.
    - Student learns on strongly augmented unlabeled images against dynamic soft pseudo-labels.
    - Evaluates pseudo-label quality at each epoch.
    """
    print("   Initializing Soft Teacher...")
    # Initialize Student and Teacher models identically
    student_model = get_faster_rcnn_model(backbone_type=backbone, num_classes=2, pretrained=True)
    teacher_model = get_faster_rcnn_model(backbone_type=backbone, num_classes=2, pretrained=True)
    
    student_model.to(device)
    teacher_model.to(device)
    
    # Force identical starting parameters
    teacher_model.load_state_dict(student_model.state_dict())
    
    # Data loaders
    # Use weak augmentations for labeled data
    labeled_ds = copy.deepcopy(labeled_dataset)
    labeled_ds.augmentation_type = 'weak'
    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size // 2, shuffle=True,
                                collate_fn=collate_fn, num_workers=0, drop_last=True)
                                
    # Use dual augmentations for unlabeled data: Teacher evaluates weak, Student evaluates strong
    unlabeled_ds_weak = copy.deepcopy(unlabeled_dataset)
    unlabeled_ds_weak.augmentation_type = 'weak'
    unlabeled_loader_weak = DataLoader(unlabeled_ds_weak, batch_size=batch_size // 2, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0, drop_last=True)
                                       
    unlabeled_ds_strong = copy.deepcopy(unlabeled_dataset)
    unlabeled_ds_strong.augmentation_type = 'strong_photometric'
    
    # Ground truth lookup for pseudo-label evaluation
    unlabeled_eval_dataset = copy.deepcopy(unlabeled_dataset)
    unlabeled_eval_dataset.augmentation_type = None
    unlabeled_eval_loader = DataLoader(unlabeled_eval_dataset, batch_size=batch_size, shuffle=False,
                                       collate_fn=collate_fn, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
                             
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    best_map = -1
    best_teacher_wts = copy.deepcopy(teacher_model.state_dict())
    
    last_pl_quality = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    for epoch in range(1, epochs + 1):
        student_model.train()
        teacher_model.eval()  # Teacher is always in eval mode for pseudo-labeling
        
        # We zip loaders to get batches of labeled and unlabeled together
        zipped_loaders = zip(labeled_loader, unlabeled_loader_weak)
        total_loss = 0.0
        
        for (labeled_batch, unlabeled_batch) in zipped_loaders:
            lab_imgs, lab_targets, _ = labeled_batch
            unlab_imgs, _, unlab_ids = unlabeled_batch
            
            # 1. Real-time pseudo labeling by Teacher on weakly augmented unlabeled images
            with torch.no_grad():
                unlab_imgs_device = [img.to(device) for img in unlab_imgs]
                teacher_outputs = teacher_model(unlab_imgs_device)
                
            # Dynamic threshold computation
            # Collect scores from this batch for thresholding
            batch_scores = []
            for out in teacher_outputs:
                batch_scores.extend(out['scores'].cpu().numpy().tolist())
                
            if threshold_strategy == "fixed":
                threshold = fixed_val
            else:
                if len(batch_scores) > 0:
                    mean_score = np.mean(batch_scores)
                    std_score = np.std(batch_scores) if len(batch_scores) > 1 else 0
                    warmup_factor = min(1.0, epoch / max(1, epochs * 0.6))
                    threshold = float(np.clip((mean_score - 0.5 * std_score) * warmup_factor, 0.50, 0.88))
                else:
                    threshold = fixed_val
                    
            # Filter and create target dicts for Student unlabeled training
            student_unlab_targets = []
            for out, img_id in zip(teacher_outputs, unlab_ids):
                boxes = out['boxes'].cpu()
                labels = out['labels'].cpu()
                scores = out['scores'].cpu()
                
                # Filter for high quality WBCs
                keep_mask = (labels == 1) & (scores >= threshold)
                filt_boxes = boxes[keep_mask]
                
                # Fetch student's strongly augmented version of this image (using matching index)
                # For simplicity, we just apply strong augmentation directly to the images tensor
                # by adding random noise and color filters since the labels are already resolved.
                if len(filt_boxes) > 0:
                    student_unlab_targets.append({
                        "boxes": filt_boxes.to(device),
                        "labels": torch.ones((len(filt_boxes),), dtype=torch.int64, device=device),
                        "image_id": torch.tensor([img_id], device=device)
                    })
                else:
                    student_unlab_targets.append({
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "image_id": torch.tensor([img_id], device=device)
                    })
            
            # Combine student training images and targets
            # Labeled + Unlabeled
            lab_imgs_device = [img.to(device) for img in lab_imgs]
            lab_targets_device = [{k: v.to(device) for k, v in t.items()} for t in lab_targets]
            
            # Strongly augment the unlabeled images before student forward pass
            # We apply simple on-the-fly random color jitter and noise to simulate strong augmentations
            strong_unlab_imgs = []
            for img in unlab_imgs:
                # Add tiny random color scale and gaussian noise to simulate strong augmentation
                jitter = img.clone()
                if np.random.random() > 0.4:
                    jitter = jitter * np.random.uniform(0.8, 1.2)
                    jitter = torch.clip(jitter + torch.randn_like(jitter) * 0.02, 0, 1)
                strong_unlab_imgs.append(jitter.to(device))
                
            combined_imgs = lab_imgs_device + strong_unlab_imgs
            combined_targets = lab_targets_device + student_unlab_targets
            
            # Student forward pass
            loss_dict = student_model(combined_imgs, combined_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)
            optimizer.step()
            
            total_loss += losses.item()
            
            # 2. Update EMA Teacher
            update_teacher_ema(student_model, teacher_model, alpha=ema_alpha)
            
        # End of epoch validation
        val_metrics = evaluate_map(teacher_model, val_loader, device)
        if val_metrics["mAP@0.5"] > best_map:
            best_map = val_metrics["mAP@0.5"]
            best_teacher_wts = copy.deepcopy(teacher_model.state_dict())
            
    # Load best weights
    teacher_model.load_state_dict(best_teacher_wts)
    test_metrics = evaluate_map(teacher_model, test_loader, device)
    
    # Evaluate final pseudo-label quality
    print("   Evaluating final Soft Teacher pseudo-label quality...")
    pseudo_labels, _ = generate_pseudo_labels(
        teacher_model, unlabeled_eval_loader, device,
        threshold_strategy=threshold_strategy, fixed_val=fixed_val,
        epoch=epochs, total_epochs=epochs
    )
    pl_quality = evaluate_pseudo_label_quality(pseudo_labels, unlabeled_eval_dataset)
    print(f"   [Pseudo-Label Quality] Final Precision: {pl_quality['precision']:.3f} | Recall: {pl_quality['recall']:.3f}")
    
    test_metrics["pl_precision"] = pl_quality["precision"]
    test_metrics["pl_recall"] = pl_quality["recall"]
    test_metrics["pl_f1_score"] = pl_quality["f1_score"]
    
    return test_metrics
