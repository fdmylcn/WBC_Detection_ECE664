import os
import json
import torch
import subprocess
import sys
from dataset import get_dataset_splits, get_labeled_unlabeled_split, WBCDataset
from train import run_supervised_training, run_stac_training, run_soft_teacher_training

# --- Configuration ---
# Mode: "real" to train on the actual TXL-PBC dataset, or "synthetic" for simulated testing
DATASET_MODE = "real"

# Path to the real TXL-PBC dataset directory
DATASET_DIR = "d:/assignment/TXL-PBC_Dataset/TXL-PBC"
COCO_PATH = os.path.join(DATASET_DIR, "data.yaml")

# Set QUICK_RUN = True for a rapid end-to-end execution of all 5 methods and 4 fractions (under 3 minutes)
# Set QUICK_RUN = False for full training (suitable for GPU execution)
QUICK_RUN = False

# Backbone: "mobilenet_v3" (highly optimized for CPU/speed) or "resnet50" (standard, slower)
BACKBONE = "mobilenet_v3"

def main():
    print("=" * 80)
    print("  SEMI-SUPERVISED WBC DETECTION COMPARATIVE BENCHMARK RUNNER")
    print("=" * 80)
    
    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Adjust Hyperparameters based on execution mode
    if QUICK_RUN:
        print("[Mode] RUNNING IN QUICK-DEBUG MODE (fast execution)")
        num_synthetic = 60
        img_size = (224, 224)
        supervised_epochs = 2
        stac_teacher_epochs = 2
        stac_student_epochs = 2
        soft_epochs = 2
        batch_size = 4
    else:
        print("[Mode] RUNNING IN PRODUCTION MODE (slower, standard dataset and epochs)")
        num_synthetic = 300
        img_size = (416, 416)
        supervised_epochs = 15
        stac_teacher_epochs = 10
        stac_student_epochs = 15
        soft_epochs = 15
        batch_size = 4
        
    # 3. Check and configure Dataset Mode
    if DATASET_MODE == "real":
        if not os.path.exists(DATASET_DIR):
            print(f"\n[WARNING] Real TXL-PBC dataset directory not found at: {DATASET_DIR}")
            print(f"          Please place your TXL-PBC dataset folder at: {DATASET_DIR}")
            print("          Falling back to 'synthetic' mode for immediate demonstration purposes...")
            mode = "synthetic"
        else:
            mode = "real"
            print(f"[Dataset] Loading real TXL-PBC dataset from: {DATASET_DIR}")
    else:
        mode = "synthetic"
        print("[Dataset] Running in synthetic mode...")

    # Create Dataset Splits (70% Train, 15% Val, 15% Test)
    train_ids, val_ids, test_ids = get_dataset_splits(
        mode=mode,
        coco_path=COCO_PATH,
        num_synthetic=num_synthetic, 
        seed=42, 
        img_size=img_size
    )
    
    print(f"[Splits] Train set size: {len(train_ids)} images")
    print(f"[Splits] Val set size:   {len(val_ids)} images")
    print(f"[Splits] Test set size:  {len(test_ids)} images")
    
    # Validation and Test datasets (no augmentations)
    val_dataset = WBCDataset(mode=mode, dataset_dir=DATASET_DIR, split_type="val", img_ids=val_ids, transform=None, augmentation_type=None, img_size=img_size)
    test_dataset = WBCDataset(mode=mode, dataset_dir=DATASET_DIR, split_type="test", img_ids=test_ids, transform=None, augmentation_type=None, img_size=img_size)
    
    # Fractions to test: 1%, 5%, 10%, 20%
    label_fractions = [0.01, 0.05, 0.10, 0.20]
    
    # Result storage structure
    results = {
        "supervised": {},
        "stac_fixed": {},
        "stac_adaptive": {},
        "soft_teacher_fixed": {},
        "soft_teacher_adaptive": {}
    }
    
    # 4. Benchmarking Loops
    for fraction in label_fractions:
        frac_pct = int(fraction * 100)
        frac_str = f"{frac_pct}%"
        
        print("\n" + "#" * 50)
        print(f"### RUNNING SCENARIO: LABEL FRACTION = {frac_str} ###")
        print("#" * 50)
        
        # Partition Train Split into Labeled and Unlabeled Pools
        labeled_ids, unlabeled_ids = get_labeled_unlabeled_split(train_ids, fraction, seed=42)
        print(f"[Data Partition] Labeled Pool Size:   {len(labeled_ids)} images")
        print(f"[Data Partition] Unlabeled Pool Size: {len(unlabeled_ids)} images")
        
        # Create pool datasets
        labeled_dataset = WBCDataset(mode=mode, dataset_dir=DATASET_DIR, split_type="train", img_ids=labeled_ids, transform=None, augmentation_type='weak', img_size=img_size)
        unlabeled_dataset = WBCDataset(mode=mode, dataset_dir=DATASET_DIR, split_type="train", img_ids=unlabeled_ids, transform=None, augmentation_type='strong', img_size=img_size)
        
        # --- Method 1: Supervised Baseline ---
        print("\n--- Method 1: Faster R-CNN (Supervised Baseline) ---")
        metrics = run_supervised_training(
            labeled_dataset, val_dataset, test_dataset, device,
            backbone=BACKBONE, epochs=supervised_epochs, batch_size=batch_size
        )
        results["supervised"][frac_str] = metrics
        print(f"    [Results] Test mAP@0.5: {metrics['mAP@0.5']:.4f} | Recall: {metrics['recall']:.4f}")
        
        # --- Method 2: STAC (Fixed Threshold) ---
        print("\n--- Method 2: STAC (Fixed Threshold = 0.85) ---")
        metrics = run_stac_training(
            labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
            backbone=BACKBONE, teacher_epochs=stac_teacher_epochs, student_epochs=stac_student_epochs,
            batch_size=batch_size, threshold_strategy="fixed", fixed_val=0.85
        )
        results["stac_fixed"][frac_str] = metrics
        print(f"    [Results] Test mAP@0.5: {metrics['mAP@0.5']:.4f} | PL F1: {metrics['pl_f1_score']:.4f}")
        
        # --- Method 3: STAC (Adaptive Threshold) ---
        print("\n--- Method 3: STAC (Adaptive Threshold) ---")
        metrics = run_stac_training(
            labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
            backbone=BACKBONE, teacher_epochs=stac_teacher_epochs, student_epochs=stac_student_epochs,
            batch_size=batch_size, threshold_strategy="adaptive"
        )
        results["stac_adaptive"][frac_str] = metrics
        print(f"    [Results] Test mAP@0.5: {metrics['mAP@0.5']:.4f} | PL F1: {metrics['pl_f1_score']:.4f}")
        
        # --- Method 4: Soft Teacher (Fixed Threshold) ---
        print("\n--- Method 4: Soft Teacher (Fixed Threshold = 0.85) ---")
        metrics = run_soft_teacher_training(
            labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
            backbone=BACKBONE, epochs=soft_epochs, batch_size=batch_size,
            threshold_strategy="fixed", fixed_val=0.85
        )
        results["soft_teacher_fixed"][frac_str] = metrics
        print(f"    [Results] Test mAP@0.5: {metrics['mAP@0.5']:.4f} | PL F1: {metrics['pl_f1_score']:.4f}")
        
        # --- Method 5: Soft Teacher (Adaptive Threshold) ---
        print("\n--- Method 5: Soft Teacher (Adaptive Threshold) ---")
        metrics = run_soft_teacher_training(
            labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, device,
            backbone=BACKBONE, epochs=soft_epochs, batch_size=batch_size,
            threshold_strategy="adaptive"
        )
        results["soft_teacher_adaptive"][frac_str] = metrics
        print(f"    [Results] Test mAP@0.5: {metrics['mAP@0.5']:.4f} | PL F1: {metrics['pl_f1_score']:.4f}")

    # 5. Save all results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Results saved to results.json")
    print("=" * 80)
    
    # 6. Call plotting script to generate figures
    print("[Plots] Generating comparison figures and tables...")
    try:
        subprocess.run([sys.executable, "plot_results.py"], check=True)
        print("[Plots] Done. Review 'sensitivity_comparison.png' and 'pseudo_label_quality.png'")
    except Exception as e:
        print(f"[Error] Failed to run plot_results.py automatically: {e}")

if __name__ == "__main__":
    main()
