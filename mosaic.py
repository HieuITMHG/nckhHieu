import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os
import random
import glob

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

# Hàm tạo mosaic từ 4 ảnh
def create_mosaic(images, bboxes_list, output_size=(1280, 1280)):
    yc, xc = output_size[0] // 2, output_size[1] // 2  # Kích thước mỗi góc: (640, 640)
    h, w = output_size  # Kích thước ảnh mosaic: (1280, 1280)
    
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_bboxes = []
    mosaic_labels = []
    
    for idx, (img, bboxes) in enumerate(zip(images, bboxes_list)):
        orig_h, orig_w = img.shape[:2]  # Kích thước gốc của ảnh
        resized_img = cv2.resize(img, (xc, yc))  # Resize về kích thước góc
        
        # Gán vào mosaic_img
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
        
        # Điều chỉnh bounding box
        scale_x = xc / orig_w  # Tỷ lệ resize theo chiều rộng
        scale_y = yc / orig_h  # Tỷ lệ resize theo chiều cao
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, label = bbox
            # Điều chỉnh tọa độ theo tỷ lệ resize
            x_min_new = x_min * scale_x + x_shift
            y_min_new = y_min * scale_y + y_shift
            x_max_new = x_max * scale_x + x_shift
            y_max_new = y_max * scale_y + y_shift
            
            # Kiểm tra bounding box hợp lệ
            if x_max_new > x_min_new and y_max_new > y_min_new and x_max_new <= w and y_max_new <= h:
                # Chuyển sang định dạng [x_center, y_center, width, height]
                width = x_max_new - x_min_new
                height = y_max_new - y_min_new
                x_center = x_min_new + width / 2
                y_center = y_min_new + height / 2
                mosaic_bboxes.append([x_center, y_center, width, height])
                mosaic_labels.append(label)
    
    return mosaic_img, mosaic_bboxes, mosaic_labels

# Hàm hiển thị ảnh với bounding box
def draw_bboxes(img, bboxes, labels, title="Image"):
    img_copy = img.copy()
    for bbox, label in zip(bboxes, labels):
        x_center, y_center, width, height = map(int, bbox)
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_copy, str(label), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()

# Hàm lưu ảnh và ground truth
def save_output(mosaic_img, mosaic_bboxes, mosaic_labels, img_path, txt_file, frame_id):
    cv2.imwrite(img_path, cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR))
    with open(txt_file, 'a') as f:
        for bbox, label in zip(mosaic_bboxes, mosaic_labels):
            x_center, y_center, width, height = map(int, bbox)
            confidence_score = 1.0
            f.write(f"{frame_id} {x_center} {y_center} {width} {height} {confidence_score}\n")

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

# Thu thập tất cả ảnh và ground truth
folders = sorted([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
all_image_paths = []
all_bboxes = []
for folder in folders:
    img_dir = os.path.join(train_path, folder, "img1")
    gt_file = os.path.join(train_path, folder, "gt", "gt.txt")
    
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    all_image_paths.extend(image_files)
    
    gt_dict = {}
    if os.path.exists(gt_file):
        gt_dict = parse_ground_truth(gt_file)
    
    for img_path in image_files:
        frame_id = int(os.path.basename(img_path).replace(".jpg", ""))
        bboxes = gt_dict.get(frame_id, [])
        all_bboxes.append(bboxes)

# Tạo nhiều ảnh mosaic (ví dụ: 10 ảnh)
num_mosaics = 10
mosaic_counter = 1

for _ in range(num_mosaics):
    if len(all_image_paths) < 4:
        print("Không đủ ảnh để tạo mosaic")
        break
    
    # Chọn ngẫu nhiên 4 ảnh
    selected_indices = random.sample(range(len(all_image_paths)), 4)
    images = []
    bboxes_list = []
    
    for idx in selected_indices:
        img_path = all_image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh {img_path}, bỏ qua")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        bboxes_list.append(all_bboxes[idx])
    
    if len(images) < 4:
        print(f"Không đủ 4 ảnh cho mosaic {mosaic_counter}, bỏ qua")
        continue
    
    # Tạo mosaic
    mosaic_img, mosaic_bboxes, mosaic_labels = create_mosaic(images, bboxes_list)
    
    # Tạo tên file đầu ra
    output_frame_id = f"{mosaic_counter:06d}"
    output_img_path = os.path.join(img_output_path, f"{output_frame_id}.jpg")
    
    # Hiển thị kết quả
    draw_bboxes(mosaic_img, mosaic_bboxes, mosaic_labels, title=f"Mosaic Image {output_frame_id}")
    
    # Lưu ảnh và ground truth
    save_output(mosaic_img, mosaic_bboxes, mosaic_labels, output_img_path, gt_output_file, output_frame_id)
    
    mosaic_counter += 1