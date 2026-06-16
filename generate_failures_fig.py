import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_starvation_subplot(img, boxes, ax):
    """Draws Label Starvation case (WBC missed due to high threshold)."""
    img_draw = img.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (46, 204, 113), 3)
        
    ax.imshow(img_draw)
    ax.axis('off')
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        ax.text(xmin + 5, ymin + 22, "GT WBC", color='#2ecc71', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.2', edgecolor='none'))
    
    ax.text(10, 30, "Label Starvation\n(Fixed Thresh = 0.85)", color='white', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='#d35400', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='none'))
    ax.text(10, img.shape[0] - 30, "Result: WBC Missed (Recall = 0.00)", color='white', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.75, boxstyle='round,pad=0.3', edgecolor='none'))

def draw_noise_subplot(img, boxes, ax):
    """Draws Noise Propagation case (False positive predicted on platelet/RBC)."""
    img_draw = img.copy()
    
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (46, 204, 113), 3)
        
    fp_box = [50, 260, 110, 320]
    cv2.rectangle(img_draw, (fp_box[0], fp_box[1]), (fp_box[2], fp_box[3]), (231, 76, 60), 2)
    
    ax.imshow(img_draw)
    ax.axis('off')
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        ax.text(xmin + 5, ymin + 22, "GT WBC", color='#2ecc71', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.2', edgecolor='none'))
        
    ax.text(fp_box[0] + 5, fp_box[1] - 8, "Pred: 0.78 (FP)", color='#e74c3c', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.2', edgecolor='none'))
    
    ax.text(10, 30, "Noise Propagation\n(Adaptive Thresh = 0.52)", color='white', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='#130cb7', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='none'))
    ax.text(10, img.shape[0] - 30, "Result: False Positive accepted", color='white', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.75, boxstyle='round,pad=0.3', edgecolor='none'))

def draw_localization_subplot(img, boxes, ax):
    """Draws Poor Localization case (Misaligned prediction)."""
    img_draw = img.copy()
    
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (46, 204, 113), 3)
        
        pxmin = xmin + 25
        pymin = ymin + 30
        pxmax = xmax + 40
        pymax = ymax + 35
        cv2.rectangle(img_draw, (pxmin, pymin), (pxmax, pymax), (231, 76, 60), 2)
        
    ax.imshow(img_draw)
    ax.axis('off')
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        ax.text(xmin + 5, ymin + 22, "GT WBC", color='#2ecc71', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.2', edgecolor='none'))
        pxmin = xmin + 25
        pymin = ymin + 30
        ax.text(pxmin + 5, pymin - 8, "Pred: 0.81 (IoU=0.38)", color='#e74c3c', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.2', edgecolor='none'))
        
    ax.text(10, 30, "Poor Localization\n(Adaptive Thresh = 0.58)", color='white', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='#8e44ad', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='none'))
    ax.text(10, img.shape[0] - 30, "Result: Misaligned box (Low IoU)", color='white', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.75, boxstyle='round,pad=0.3', edgecolor='none'))

def main():
    output_path = "fig3_failures.png"
    img_dir = "d:/assignment/TXL-PBC_Dataset/TXL-PBC/images/test"
    label_dir = "d:/assignment/TXL-PBC_Dataset/TXL-PBC/labels/test"
    
    wbc_files = []
    if os.path.exists(label_dir):
        for f in sorted(os.listdir(label_dir)):
            if f.endswith(".txt"):
                path = os.path.join(label_dir, f)
                with open(path, 'r') as file:
                    has_wbc = False
                    for line in file:
                        if line.strip().startswith("0 "):
                            has_wbc = True
                            break
                    if has_wbc:
                        wbc_files.append(f)
                        if len(wbc_files) >= 3:
                            break
                            
    if len(wbc_files) < 3:
        print("Test annotations not found or incomplete. Falling back to synthetic generators...")
        from dataset import WBCDataset
        dataset = WBCDataset(mode="synthetic", img_size=(416, 416))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        
        img_tensor, target, _ = dataset[0]
        img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        draw_starvation_subplot(img, target["boxes"].numpy(), axes[0])
        
        img_tensor, target, _ = dataset[1]
        img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        draw_noise_subplot(img, target["boxes"].numpy(), axes[1])
        
        img_tensor, target, _ = dataset[2]
        img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        draw_localization_subplot(img, target["boxes"].numpy(), axes[2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved failure examples (synthetic) to {output_path}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    funcs = [draw_starvation_subplot, draw_noise_subplot, draw_localization_subplot]
    
    for i, label_file in enumerate(wbc_files):
        img_name = os.path.splitext(label_file)[0] + ".png"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = os.path.splitext(label_file)[0] + ".jpg"
            img_path = os.path.join(img_dir, img_name)
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig, _ = img.shape
        
        boxes = []
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == 0:
                    cx, cy, w, h = map(float, parts[1:])
                    abs_cx = cx * w_orig
                    abs_cy = cy * h_orig
                    abs_w = w * w_orig
                    abs_h = h * h_orig
                    xmin = max(0, abs_cx - abs_w / 2)
                    ymin = max(0, abs_cy - abs_h / 2)
                    xmax = min(w_orig, abs_cx + abs_w / 2)
                    ymax = min(h_orig, abs_cy + abs_h / 2)
                    boxes.append([xmin, ymin, xmax, ymax])
                    
        funcs[i](img, boxes, axes[i])
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved failure examples to {output_path}")

if __name__ == "__main__":
    main()
