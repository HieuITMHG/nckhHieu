import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Hàm đọc ground truth từ file gt.txt
def parse_ground_truth(gt_file):
    gt_dict = {}
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 11:
                    print(f"Cảnh báo: Dòng không hợp lệ trong {gt_file}: {line.strip()}")
                    continue
                try:
                    frame_id, obj_id, x, y, w, h, _, _, _, _, label = map(int, parts)
                    if frame_id not in gt_dict:
                        gt_dict[frame_id] = []
                    gt_dict[frame_id].append([x, y, x + w, y + h, label])  # [x_min, y_min, x_max, y_max, label]
                except ValueError as e:
                    print(f"Lỗi khi phân tích dòng trong {gt_file}: {line.strip()} - {e}")
    except FileNotFoundError:
        print(f"Không tìm thấy file ground truth: {gt_file}")
    return gt_dict

# Hàm mix-up
def mix_up(image1, image2, bboxes1, bboxes2, alpha, target_size=(640, 640)):
    
    # Resize image2 to match image1's dimensions
    image2 = cv2.resize(image2, target_size, interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, target_size, interpolation=cv2.INTER_AREA)
    
    # Generate mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Mix images
    mixed_img = Image.fromarray(np.uint8(lam * np.array(image1) + (1 - lam) * np.array(image2)))
    mixed_img = np.array(mixed_img)
    
    # Mix bounding boxes (keep all boxes from both images)
    mixed_bboxes = []
    mixed_labels = []
    
    # Scale bboxes2 to match image1's dimensions
    scale_x2 = target_size[0] / image2.shape[1]
    scale_y2 = target_size[1] / image2.shape[0]

    scale_x1 = target_size[0] / image1.shape[1]
    scale_y1 = target_size[1] / image1.shape[0]
    
    # Add bboxes from image1
    for bbox in bboxes1:
        x_min, y_min, x_max, y_max, label = bbox
        x_min_new = x_min * scale_x1
        y_min_new = y_min * scale_y1
        x_max_new = x_max * scale_x1
        y_max_new = y_max * scale_y1
        if x_max_new > x_min_new and y_max_new > y_min_new:
            mixed_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
            mixed_labels.append(label)
    
    # Add scaled bboxes from image2
    for bbox in bboxes2:
        x_min, y_min, x_max, y_max, label = bbox
        x_min_new = x_min * scale_x2
        y_min_new = y_min * scale_y2
        x_max_new = x_max * scale_x2
        y_max_new = y_max * scale_y2
        if x_max_new > x_min_new and y_max_new > y_min_new:
            mixed_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])
            mixed_labels.append(label)
    
    return mixed_img, mixed_bboxes, mixed_labels

# Hàm tạo mosaic từ 4 ảnh
def create_mosaic(images, bboxes_list, output_size=(1280, 1280)):
    try:
        yc, xc = output_size[0] // 2, output_size[1] // 2
        h, w = output_size
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        for idx, (img, bboxes) in enumerate(zip(images, bboxes_list)):
            if img is None:
                print(f"Ảnh không hợp lệ tại vị trí {idx}")
                continue
            orig_h, orig_w = img.shape[:2]
            resized_img = cv2.resize(img, (xc, yc))
            
            if idx == 0:  # Top-left
                mosaic_img[0:yc, 0:xc] = resized_img
                x_shift, y_shift = 0, 0
            elif idx == 1:  # Top-right
                mosaic_img[0:yc, xc:w] = resized_img
                x_shift, y_shift = xc, 0
            elif idx == 2:  # Bottom-left
                mosaic_img[yc:h, 0:xc] = resized_img
                x_shift, y_shift = 0, yc
            elif idx == 3:  # Bottom-right
                mosaic_img[yc:h, xc:w] = resized_img
                x_shift, y_shift = xc, yc
            
            scale_x = xc / orig_w
            scale_y = yc / orig_h
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, label = bbox
                x_min_new = x_min * scale_x + x_shift
                y_min_new = y_min * scale_y + y_shift
                x_max_new = x_max * scale_x + x_shift
                y_max_new = y_max * scale_y + y_shift
                if x_max_new > x_min_new and y_max_new > y_min_new and x_max_new <= w and y_max_new <= h:
                    width = x_max_new - x_min_new
                    height = y_max_new - y_min_new
                    x_center = x_min_new + width / 2
                    y_center = y_min_new + height / 2
                    mosaic_bboxes.append([x_center, y_center, width, height])
                    mosaic_labels.append(label)
        
        max_start_x = w - xc
        max_start_y = h - yc
        start_x = random.randint(0, max_start_x)
        start_y = random.randint(0, max_start_y)
        cropped_mosaic = mosaic_img[start_y:start_y + yc, start_x:start_x + xc]
        
        cropped_bboxes = []
        cropped_labels = []
        for bbox, label in zip(mosaic_bboxes, mosaic_labels):
            x_center, y_center, width, height = bbox
            x_center_new = x_center - start_x
            y_center_new = y_center - start_y
            x_min_new = x_center_new - width / 2
            y_min_new = y_center_new - height / 2
            x_max_new = x_center_new + width / 2
            y_max_new = y_center_new + height / 2
            if (x_max_new > 0 and x_min_new < xc and y_max_new > 0 and y_min_new < yc):
                x_min_new = max(0, x_min_new)
                y_min_new = max(0, y_min_new)
                x_max_new = min(xc, x_max_new)
                y_max_new = min(yc, y_max_new)
                width_new = x_max_new - x_min_new
                height_new = y_max_new - y_min_new
                if width_new > 0 and height_new > 0:
                    x_center_new = x_min_new + width_new / 2
                    y_center_new = y_min_new + height_new / 2
                    cropped_bboxes.append([x_center_new, y_center_new, width_new, height_new])
                    cropped_labels.append(label)
        
        return cropped_mosaic, cropped_bboxes, cropped_labels
    except Exception as e:
        print(f"Lỗi trong create_mosaic: {e}")
        return None, [], []

# Hàm hiển thị ảnh với bounding box
def draw_bboxes(img, bboxes, labels, title="Image"):
    try:
        if img is None:
            print(f"Không thể hiển thị ảnh: {title}")
            return
        img_copy = img.copy()
        for bbox, label in zip(bboxes, labels):
            if len(bbox) == 4:  # Center format
                x_center, y_center, width, height = map(int, bbox)
                x_min = x_center - width // 2
                y_min = y_center - height // 2
                x_max = x_center + width // 2
                y_max = y_center + height // 2
            else:  # Corner format
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img_copy, str(label), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(img_copy)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh {title}: {e}")

# Hàm lưu ảnh và ground truth
def save_output(img, bboxes, labels, img_path, txt_file, frame_id, bbox_format='center'):
    try:
        if img is None:
            print(f"Không thể lưu ảnh: {img_path}")
            return
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(txt_file, 'a') as f:
            for bbox, label in zip(bboxes, labels):
                if bbox_format == 'center':
                    x_center, y_center, width, height = map(int, bbox)
                    f.write(f"{frame_id} {x_center} {y_center} {width} {height} 1.0 {label}\n")
                else:  # corner
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    f.write(f"{frame_id} {x_min} {y_min} {x_max - x_min} {y_max - y_min} 1.0 {label}\n")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh hoặc ground truth {img_path}: {e}")

# Đường dẫn gốc
base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "train")
output_path = os.path.join(base_path, "personpath_after_aug")
img_output_path = os.path.join(output_path, "imgs")
gt_output_path = os.path.join(output_path, "gt")

# Tạo thư mục đầu ra
os.makedirs(img_output_path, exist_ok=True)
os.makedirs(gt_output_path, exist_ok=True)
gt_output_file = os.path.join(gt_output_path, "gt.txt")

# Xóa file gt.txt cũ nếu tồn tại
if os.path.exists(gt_output_file):
    os.remove(gt_output_file)

# Thu thập ảnh với bước nhảy 30
folders = sorted([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
all_image_paths = []
all_bboxes = []
for folder in folders:
    img_dir = os.path.join(train_path, folder, "img1")
    gt_file = os.path.join(train_path, folder, "gt", "gt.txt")
    
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    # Lấy ảnh với bước nhảy 30
    image_files = image_files[::30]
    all_image_paths.extend(image_files)
    
    gt_dict = parse_ground_truth(gt_file) if os.path.exists(gt_file) else {}
    
    for img_path in image_files:
        frame_id = int(os.path.basename(img_path).split('.')[0])
        bboxes = gt_dict.get(frame_id, [])
        all_bboxes.append(bboxes)

# Tạo ảnh mosaic và mix-up
mosaic_counter = 1
while len(all_image_paths) >= 4:
    # Chọn ngẫu nhiên 4 ảnh
    selected_indices = random.sample(range(len(all_image_paths)), 4)
    images = []
    bboxes_list = []
    
    # Đọc ảnh và bounding box
    for idx in selected_indices:
        img_path = all_image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh {img_path}, bỏ qua")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        bboxes_list.append(all_bboxes[idx])
    
    # Kiểm tra số lượng ảnh hợp lệ
    if len(images) < 4:
        print(f"Không đủ 4 ảnh cho mosaic {mosaic_counter}, bỏ qua")
        continue
    
    # Tạo mosaic
    mosaic_img, mosaic_bboxes, mosaic_labels = create_mosaic(images, bboxes_list)
    if mosaic_img is None:
        print(f"Không thể tạo mosaic {mosaic_counter}, bỏ qua")
        continue
    
    # Tạo mix-up từ hai ảnh đầu tiên
    mixup_img, mixup_bboxes, mixup_labels = mix_up(images[0], images[1], bboxes_list[0], bboxes_list[1], 0.5, (640, 640))
    if mixup_img is None:
        print(f"Không thể tạo mix-up {mosaic_counter}, bỏ qua")
        continue
    
    # Tạo tên file đầu ra
    output_frame_id = f"{mosaic_counter:06d}"
    output_img_path_mosaic = os.path.join(img_output_path, f"mosaic_{output_frame_id}.jpg")
    output_img_path_orig1 = os.path.join(img_output_path, f"orig1_{output_frame_id}.jpg")
    output_img_path_orig2 = os.path.join(img_output_path, f"orig2_{output_frame_id}.jpg")
    output_img_path_mixup = os.path.join(img_output_path, f"mixup_{output_frame_id}.jpg")
    
    # Hiển thị kết quả
    draw_bboxes(images[0], bboxes_list[0], [bbox[-1] for bbox in bboxes_list[0]], title=f"Original Image 1 {output_frame_id}")
    draw_bboxes(images[1], bboxes_list[1], [bbox[-1] for bbox in bboxes_list[1]], title=f"Original Image 2 {output_frame_id}")
    draw_bboxes(mixup_img, mixup_bboxes, mixup_labels, title=f"Mix-up Image {output_frame_id}")
    draw_bboxes(mosaic_img, mosaic_bboxes, mosaic_labels, title=f"Mosaic Image {output_frame_id}")
    
    # Lưu ảnh và ground truth
    save_output(images[0], bboxes_list[0], [bbox[-1] for bbox in bboxes_list[0]], output_img_path_orig1, gt_output_file, f"orig1_{output_frame_id}", bbox_format='corner')
    save_output(images[1], bboxes_list[1], [bbox[-1] for bbox in bboxes_list[1]], output_img_path_orig2, gt_output_file, f"orig2_{output_frame_id}", bbox_format='corner')
    save_output(mixup_img, mixup_bboxes, mixup_labels, output_img_path_mixup, gt_output_file, f"mixup_{output_frame_id}", bbox_format='corner')
    save_output(mosaic_img, mosaic_bboxes, mosaic_labels, output_img_path_mosaic, gt_output_file, f"mosaic_{output_frame_id}", bbox_format='center')
    
    # Xóa các ảnh đã sử dụng
    selected_indices.sort(reverse=True)
    for idx in selected_indices:
        all_image_paths.pop(idx)
        all_bboxes.pop(idx)
    
    mosaic_counter += 1

# Thông báo khi hoàn thành
print(f"Đã tạo {mosaic_counter - 1} bộ ảnh (mosaic, mix-up, gốc).")