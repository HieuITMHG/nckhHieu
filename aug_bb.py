import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Hàm đọc ground truth từ file gt.txt
def parse_ground_truth(gt_file):
    gt_dict = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = parts[0]  # frame_id là chuỗi (ví dụ: "000001")
            x_center, y_center, width, height, confidence = map(float, parts[1:6])
            # Chuyển từ [x_center, y_center, width, height] sang [x_min, y_min, x_max, y_max]
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            if frame_id not in gt_dict:
                gt_dict[frame_id] = []
            gt_dict[frame_id].append([x_min, y_min, x_max, y_max, 1])  # Giả định label = 1 (theo dữ liệu)
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

# Đường dẫn gốc đến thư mục personpath_after_aug
base_path = os.path.dirname(os.path.abspath(__file__))  # Lấy đường dẫn của file hiện tại
aug_path = os.path.join(base_path, "personpath_after_aug")  # Thư mục personpath_after_aug

# Đường dẫn đến thư mục ảnh và ground truth
img_dir = os.path.join(aug_path, "imgs")
gt_file = os.path.join(aug_path, "gt", "gt.txt")

# Kiểm tra xem thư mục ảnh và file gt.txt tồn tại không
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh {img_dir}")
if not os.path.exists(gt_file):
    raise FileNotFoundError(f"Không tìm thấy tệp ground truth {gt_file}")

# Đọc ground truth
gt_dict = parse_ground_truth(gt_file)

# Lấy danh sách tất cả ảnh trong thư mục imgs
image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
if not image_files:
    print(f"Không tìm thấy ảnh trong thư mục {img_dir}")
    exit()

# Duyệt qua từng ảnh
for img_path in image_files:
    frame_id = os.path.basename(img_path).replace(".jpg", "")  # Lấy frame_id từ tên file (ví dụ: "000001")
    
    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh {img_path}, bỏ qua.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    
    # Lấy bounding box và nhãn
    bboxes = []
    labels = []
    if frame_id in gt_dict:
        for bbox in gt_dict[frame_id]:
            x_min, y_min, x_max, y_max, label = bbox
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
    else:
        print(f"Không tìm thấy ground truth cho ảnh {frame_id}, hiển thị ảnh không có bounding box.")
    
    # Hiển thị ảnh với bounding box
    draw_bboxes(img, bboxes, labels, title=f"Image {frame_id} from personpath_after_aug")