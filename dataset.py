import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WBCDataset(Dataset):
    """
    Dual-mode dataset for WBC Detection, supporting real YOLO-formatted datasets
    like TXL-PBC, and synthetic cell generation for fast debugging.
    """
    def __init__(self, mode="real", dataset_dir="d:/assignment/TXL-PBC_Dataset", split_type="train", 
                 img_ids=None, transform=None, augmentation_type=None, img_size=(416, 416)):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.split_type = split_type
        self.transform = transform
        self.augmentation_type = augmentation_type  # 'weak', 'strong', or None
        self.img_size = img_size
        
        if self.mode == "real":
            self.img_dir = os.path.join(self.dataset_dir, "images", split_type)
            self.label_dir = os.path.join(self.dataset_dir, "labels", split_type)
            
            # Find and sort all images in directory to ensure consistency
            if os.path.exists(self.img_dir):
                all_files = os.listdir(self.img_dir)
                self.img_files = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:
                self.img_files = []
                print(f"[Warning] Image directory not found: {self.img_dir}")
            
            # Subsample files based on indices (used for labeled/unlabeled fraction splits)
            if img_ids is not None:
                self.img_files = [self.img_files[i] for i in img_ids if i < len(self.img_files)]
                
            self.img_ids = list(range(len(self.img_files)))
        else:
            # Synthetic Mode
            self.num_synthetic = 100 if img_ids is None else len(img_ids)
            self.img_ids = list(range(self.num_synthetic))
            
            # Generate deterministic synthetic annotations for repeatability
            self.synthetic_anns = {}
            random.seed(42)
            for idx in self.img_ids:
                # Generate between 1 and 3 WBCs per image
                num_wbc = random.randint(1, 3)
                boxes = []
                for _ in range(num_wbc):
                    w = random.randint(50, 90)
                    h = random.randint(50, 90)
                    x = random.randint(10, self.img_size[0] - w - 10)
                    y = random.randint(10, self.img_size[1] - h - 10)
                    boxes.append([x, y, w, h])
                self.synthetic_anns[idx] = boxes

    def __len__(self):
        return len(self.img_ids)

    def _generate_synthetic_image(self, idx):
        """Generates a realistic blood smear microscope image."""
        # Seed based on idx for deterministic generation
        np.random.seed(idx)
        random.seed(idx)
        
        # 1. Base background: pale yellowish-blue tint representing microscope illumination
        img = np.ones((self.img_size[1], self.img_size[0], 3), dtype=np.uint8) * 245
        img[:, :, 0] = np.random.randint(235, 245)  # B
        img[:, :, 1] = np.random.randint(240, 250)  # G
        img[:, :, 2] = np.random.randint(245, 255)  # R
        
        # Add subtle vignetting and lighting variations
        center_x, center_y = self.img_size[0] // 2, self.img_size[1] // 2
        for y in range(self.img_size[1]):
            for x in range(self.img_size[0]):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                factor = 1.0 - (dist / max(self.img_size)) * 0.12
                img[y, x] = np.clip(img[y, x] * factor, 0, 255).astype(np.uint8)

        # 2. Draw Red Blood Cells (RBCs) - pinkish-red circles with paler centers
        num_rbcs = np.random.randint(30, 60)
        for _ in range(num_rbcs):
            r = np.random.randint(18, 28)
            cx = np.random.randint(0, self.img_size[0])
            cy = np.random.randint(0, self.img_size[1])
            # Draw outer pink membrane
            color = (np.random.randint(120, 150), np.random.randint(100, 130), np.random.randint(220, 250))
            cv2.circle(img, (cx, cy), r, color, -1)
            # Draw paler biconcave center
            pale_color = (np.random.randint(180, 210), np.random.randint(170, 200), np.random.randint(235, 255))
            cv2.circle(img, (cx, cy), int(r * 0.45), pale_color, -1)

        # 3. Draw Platelets - tiny purple irregular dots
        num_platelets = np.random.randint(5, 15)
        for _ in range(num_platelets):
            px = np.random.randint(10, self.img_size[0] - 10)
            py = np.random.randint(10, self.img_size[1] - 10)
            pr = np.random.randint(2, 5)
            color = (np.random.randint(120, 160), np.random.randint(70, 90), np.random.randint(110, 140))
            # Slightly irregular shape
            pts = np.array([[px-pr, py], [px, py-pr+1], [px+pr-1, py+1], [px-1, py+pr-1]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)

        # 4. Draw White Blood Cells (WBCs) - larger blue-violet cells with dark purple nuclei
        boxes = self.synthetic_anns[idx]
        for box in boxes:
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            rx, ry = w // 2, h // 2
            
            # Cytoplasm: light blue-greyish color
            cyto_color = (np.random.randint(210, 240), np.random.randint(180, 200), np.random.randint(160, 180))
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, cyto_color, -1)
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (180, 140, 120), 2)  # border
            
            # Multi-lobed nucleus: dark violet-purple color
            n_lobes = np.random.randint(2, 5)
            nucl_color = (np.random.randint(110, 140), np.random.randint(20, 50), np.random.randint(70, 90))
            for lobe in range(n_lobes):
                # Put lobes offset from center
                lcx = cx + int(rx * 0.35 * np.cos(2 * np.pi * lobe / n_lobes))
                lcy = cy + int(ry * 0.35 * np.sin(2 * np.pi * lobe / n_lobes))
                lrx = np.random.randint(int(rx * 0.25), int(rx * 0.45))
                lry = np.random.randint(int(ry * 0.25), int(ry * 0.45))
                cv2.ellipse(img, (lcx, lcy), (lrx, lry), np.random.randint(0, 180), 0, 360, nucl_color, -1)

        # Add Gaussian noise
        noise = np.random.normal(0, 2, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img

    def _apply_strong_augmentations(self, image, bboxes):
        """Applies strong geometric and color augmentations for consistency training."""
        h, w, _ = image.shape
        img = image.copy()
        
        # 1. Color Jittering
        if random.random() > 0.3:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
        # 2. Gaussian Blur
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
        # 3. Coarse Dropout / Cutout (occlusion)
        if random.random() > 0.5:
            num_cutouts = random.randint(1, 3)
            for _ in range(num_cutouts):
                ch = random.randint(20, 60)
                cw = random.randint(20, 60)
                cx = random.randint(0, w - cw)
                cy = random.randint(0, h - ch)
                img[cy:cy+ch, cx:cx+cw] = np.random.randint(120, 140, (ch, cw, 3))
                
        # 4. Random Flips
        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        
        new_bboxes = []
        if flip_h:
            img = cv2.flip(img, 1)
        if flip_v:
            img = cv2.flip(img, 0)
            
        for bbox in bboxes:
            x, y, bw, bh = bbox
            if flip_h:
                x = w - (x + bw)
            if flip_v:
                y = h - (y + bh)
            new_bboxes.append([x, y, bw, bh])
            
        return img, new_bboxes

    def _apply_strong_photometric_augmentations(self, image, bboxes):
        """Applies strong color, blur, and noise/dropout augmentations WITHOUT spatial flips to preserve bounding box alignment."""
        h, w, _ = image.shape
        img = image.copy()
        
        # 1. Color Jittering
        if random.random() > 0.3:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
        # 2. Gaussian Blur
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
        # 3. Coarse Dropout / Cutout (occlusion)
        if random.random() > 0.5:
            num_cutouts = random.randint(1, 3)
            for _ in range(num_cutouts):
                ch = random.randint(20, 60)
                cw = random.randint(20, 60)
                cx = random.randint(0, w - cw)
                cy = random.randint(0, h - ch)
                img[cy:cy+ch, cx:cx+cw] = np.random.randint(120, 140, (ch, cw, 3))
                
        return img, bboxes

    def _apply_weak_augmentations(self, image, bboxes):
        """Applies simple spatial augmentations (horizontal flip)."""
        h, w, _ = image.shape
        img = image.copy()
        
        flip_h = random.random() > 0.5
        new_bboxes = []
        if flip_h:
            img = cv2.flip(img, 1)
            
        for bbox in bboxes:
            x, y, bw, bh = bbox
            if flip_h:
                x = w - (x + bw)
            new_bboxes.append([x, y, bw, bh])
            
        return img, new_bboxes

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        
        if self.mode == "real":
            # 1. Load real image
            file_name = self.img_files[index]
            path = os.path.join(self.img_dir, file_name)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # 2. Parse YOLO label file corresponding to this image
            label_file = os.path.splitext(file_name)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)
            
            bboxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, cx, cy, w, h = map(float, parts)
                            # In TXL-PBC YOLO annotations: Class 0 represents WBC
                            # Class 1 is RBC, Class 2 is Platelet. We keep ONLY WBC (Class 0)
                            if int(class_id) == 0:
                                # Convert normalized coordinates to absolute self.img_size coordinates
                                abs_cx = cx * self.img_size[0]
                                abs_cy = cy * self.img_size[1]
                                abs_w = w * self.img_size[0]
                                abs_h = h * self.img_size[1]
                                
                                # Convert center-based to box-based coordinates [xmin, ymin, w, h]
                                xmin = max(0, abs_cx - abs_w / 2)
                                ymin = max(0, abs_cy - abs_h / 2)
                                bboxes.append([xmin, ymin, abs_w, abs_h])
        else:
            # Synthetic Mode
            img = self._generate_synthetic_image(img_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = [list(box) for box in self.synthetic_anns[img_id]]

        # Apply augmentation strategies
        if self.augmentation_type == 'strong':
            img, bboxes = self._apply_strong_augmentations(img, bboxes)
        elif self.augmentation_type == 'strong_photometric':
            img, bboxes = self._apply_strong_photometric_augmentations(img, bboxes)
        elif self.augmentation_type == 'weak':
            img, bboxes = self._apply_weak_augmentations(img, bboxes)

        # Convert to float tensor and normalize to [0, 1]
        img_tensor = transforms.ToTensor()(img)
        
        # Construct target dictionary for Faster R-CNN
        boxes_tensor = []
        labels_tensor = []
        for box in bboxes:
            x, y, w, h = box
            # Faster R-CNN expects [xmin, ymin, xmax, ymax]
            xmin = max(0, float(x))
            ymin = max(0, float(y))
            xmax = min(self.img_size[0], float(x + w))
            ymax = min(self.img_size[1], float(y + h))
            
            # Keep only valid coordinates
            if xmax > xmin and ymax > ymin:
                boxes_tensor.append([xmin, ymin, xmax, ymax])
                labels_tensor.append(1)  # Foreground class WBC is 1 (0 background)

        if len(boxes_tensor) > 0:
            target = {
                "boxes": torch.as_tensor(boxes_tensor, dtype=torch.float32),
                "labels": torch.as_tensor(labels_tensor, dtype=torch.int64),
                "image_id": torch.tensor([img_id])
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id])
            }

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target, img_id


def get_dataset_splits(mode="real", coco_path=None, img_dir=None, num_synthetic=300, 
                       seed=42, img_size=(416, 416)):
    """
    Retrieves splits for the dataset.
    - Real Mode: Uses the official directory partitions in TXL-PBC_Dataset (images/train, images/val, images/test).
    - Synthetic Mode: Creates deterministic splits.
    """
    random.seed(seed)
    
    if mode == "real":
        # In real YOLO dataset, train, val, and test are separated inside separate subfolders.
        # We scan each folder and return standard integer indices for each directory.
        train_img_dir = os.path.join(os.path.dirname(coco_path), "images", "train")
        val_img_dir = os.path.join(os.path.dirname(coco_path), "images", "val")
        test_img_dir = os.path.join(os.path.dirname(coco_path), "images", "test")
        
        n_train = 0
        if os.path.exists(train_img_dir):
            n_train = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
        n_val = 0
        if os.path.exists(val_img_dir):
            n_val = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
        n_test = 0
        if os.path.exists(test_img_dir):
            n_test = len([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
        train_ids = list(range(n_train))
        val_ids = list(range(n_val))
        test_ids = list(range(n_test))
        
        return train_ids, val_ids, test_ids
    else:
        img_ids = list(range(num_synthetic))
        random.shuffle(img_ids)
        
        n = len(img_ids)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_ids = img_ids[:n_train]
        val_ids = img_ids[n_train:n_train + n_val]
        test_ids = img_ids[n_train + n_val:]
        
        return train_ids, val_ids, test_ids


def get_labeled_unlabeled_split(train_ids, fraction, seed=42):
    """
    Partitions the Training partition into Labeled and Unlabeled subsets.
    """
    random.seed(seed)
    sorted_ids = sorted(list(train_ids))
    random.shuffle(sorted_ids)
    
    n_labeled = max(1, int(len(sorted_ids) * fraction))
    labeled_ids = sorted_ids[:n_labeled]
    unlabeled_ids = sorted_ids[n_labeled:]
    
    return labeled_ids, unlabeled_ids


def collate_fn(batch):
    """Collation function for detection dataloaders."""
    return tuple(zip(*batch))
