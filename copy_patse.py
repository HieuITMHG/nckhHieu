import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Hàm đọc ground truth từ file gt.txt
def parse_ground_truth(gt_file):
    gt_dict = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            frame_id, obj_id, x, y, w, h, _, _, _, _, label = parts
            if frame_id not in gt_dict:
                gt_dict[frame_id] = []
            gt_dict[frame_id].append([x, y, x + w, y + h, label])
    return gt_dict

# Hàm hiển thị ảnh với bounding box
def draw_bboxes(img, bboxes, labels, title="Image"):
    img_copy = img.copy()
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = map(int, bbox)
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_copy, str(label), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()

# Hàm lưu ảnh và nhãn
def save_output(img, bboxes, labels, img_path, txt_path):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(txt_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox
            w, h = x_max - x_min, y_max - y_min
            f.write(f"{x_min} {y_min} {w} {h} {label}\n")

# Hàm copy-paste augmentation
def copy_paste_augmentation(base_img, base_bboxes, base_labels, src_images, src_bboxes_list, max_objects=5):
    result_img = base_img.copy()
    # Chỉ lấy [x_min, y_min, x_max, y_max] từ base_bboxes, bỏ label
    result_bboxes = [[x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max, _ in base_bboxes]
    result_labels = base_labels.copy()
    
    img_height, img_width = base_img.shape[:2]
    
    # Thu thập tất cả đối tượng từ các ảnh nguồn
    all_objects = []
    for src_img, src_bboxes in zip(src_images, src_bboxes_list):
        for bbox, label in zip(src_bboxes, [b[-1] for b in src_bboxes]):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            # Chỉ lấy đối tượng có label = 1 (ví dụ: "person")
            if label == 1:
                all_objects.append((src_img, x_min, y_min, x_max, y_max, label))
    
    # Chọn ngẫu nhiên tối đa max_objects để dán
    selected_objects = random.sample(all_objects, min(len(all_objects), max_objects))
    
    for src_img, x_min, y_min, x_max, y_max, label in selected_objects:
        # Cắt đối tượng từ ảnh nguồn
        obj_patch = src_img[y_min:y_max, x_min:x_max]
        obj_h, obj_w = obj_patch.shape[:2]
        if obj_h == 0 or obj_w == 0:
            continue
        
        # Chọn vị trí ngẫu nhiên để dán lên ảnh nền
        max_x = img_width - obj_w
        max_y = img_height - obj_h
        if max_x <= 0 or max_y <= 0:
            continue
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)
        
        # Dán đối tượng lên ảnh nền
        result_img[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_patch
        
        # Cập nhật bounding box mới
        new_bbox = [paste_x, paste_y, paste_x + obj_w, paste_y + obj_h]
        result_bboxes.append(new_bbox)
        result_labels.append(label)
    
    return result_img, result_bboxes, result_labels

# Đường dẫn gốc
base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "train")

# Danh sách các thư mục
folders = ["0BHL32Bw70E_95_104", "0BHL32Bw70E_221_227", "00np___nE5s_394_404", "00np___nE5s_465_471"]
fixed_frame_id = "000090"

# Đọc ảnh và ground truth
images = []
bboxes_list = []
labels_list = []
for folder in folders:
    img_dir = os.path.join(train_path, folder, "img1")
    gt_file = os.path.join(train_path, folder, "gt", "gt.txt")
    img_path = os.path.join(img_dir, f"{fixed_frame_id}.jpg")
    
    # Đọc ảnh
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    
    # Đọc ground truth
    if os.path.exists(gt_file):
        gt_dict = parse_ground_truth(gt_file)
        frame_id = int(fixed_frame_id)
        bboxes = gt_dict.get(frame_id, [])
        bboxes_list.append(bboxes)
        labels_list.append([b[-1] for b in bboxes])
    else:
        bboxes_list.append([])
        labels_list.append([])

# Áp dụng copy-paste augmentation
base_img = images[0]
base_bboxes = bboxes_list[0]
base_labels = labels_list[0]
src_images = images[1:]
src_bboxes_list = bboxes_list[1:]

result_img, result_bboxes, result_labels = copy_paste_augmentation(
    base_img, base_bboxes, base_labels, src_images, src_bboxes_list, max_objects=5
)

# Hiển thị kết quả
draw_bboxes(result_img, result_bboxes, result_labels, title="Copy-Paste Augmented Image")

# Lưu kết quả
output_img_path = os.path.join(base_path, "copy_paste_output.jpg")
output_txt_path = os.path.join(base_path, "copy_paste_ground_truth.txt")
save_output(result_img, result_bboxes, result_labels, output_img_path, output_txt_path)
print(f"Đã lưu ảnh tại: {output_img_path}")
print(f"Đã lưu nhãn tại: {output_txt_path}")