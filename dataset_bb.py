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
            parts = list(map(int, line.strip().split()))
            frame_id, obj_id, x, y, w, h, _, _, _, _, label = parts
            if frame_id not in gt_dict:
                gt_dict[frame_id] = []
            gt_dict[frame_id].append([x, y, x + w, y + h, label])
    return gt_dict

# Hàm hiển thị ảnh với bounding box
def draw_bboxes(img, bboxes, labels, title="Image"):
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy,  (640, 640))
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

# Đường dẫn gốc (base path) đến thư mục augmentation
base_path = os.path.dirname(os.path.abspath(__file__))  # Lấy đường dẫn của file hiện tại
train_path = os.path.join(base_path, "train")  # Thư mục train

# Biến tùy chỉnh để chọn thư mục và ảnh
selected_folder = "00np___nE5s_481_506"  # Chỉ định thư mục cụ thể, để trống ("") để xử lý tất cả thư mục
selected_frame_id = "000230"  # Chỉ định frame_id cụ thể, để trống ("") để xử lý tất cả ảnh

# Lấy danh sách thư mục cần xử lý
if selected_folder:
    # Nếu selected_folder được chỉ định, chỉ xử lý thư mục đó
    folders = [selected_folder] if os.path.isdir(os.path.join(train_path, selected_folder)) else []
    if not folders:
        raise FileNotFoundError(f"Thư mục {selected_folder} không tồn tại trong {train_path}")
else:
    # Nếu selected_folder rỗng, xử lý tất cả thư mục
    folders = sorted([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])

# Duyệt qua các thư mục
for folder in folders:
    # Đường dẫn đến thư mục ảnh và ground truth
    img_dir = os.path.join(train_path, folder, "img1")
    gt_file = os.path.join(train_path, folder, "gt", "gt.txt")
    
    # Kiểm tra xem thư mục ảnh tồn tại không
    if not os.path.exists(img_dir):
        print(f"Không tìm thấy thư mục ảnh {img_dir}, bỏ qua.")
        continue
    
    # Lấy danh sách ảnh
    if selected_frame_id:
        # Nếu selected_frame_id được chỉ định, chỉ xử lý ảnh đó
        img_path = os.path.join(img_dir, f"{selected_frame_id}.jpg")
        image_files = [img_path] if os.path.exists(img_path) else []
        if not image_files:
            print(f"Không tìm thấy ảnh {img_path}, bỏ qua.")
            continue
    else:
        # Nếu không chỉ định frame_id, xử lý tất cả ảnh
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {img_dir}, bỏ qua.")
            continue
    
    # Đọc ground truth
    gt_dict = {}
    if os.path.exists(gt_file):
        gt_dict = parse_ground_truth(gt_file)
    else:
        print(f"Không tìm thấy tệp ground truth {gt_file}, hiển thị ảnh không có bounding box.")
    
    # Duyệt qua từng ảnh
    for img_path in image_files:
        frame_id = int(os.path.basename(img_path).replace(".jpg", ""))  # Lấy frame_id từ tên file
        
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
        
        # Hiển thị ảnh với bounding box
        draw_bboxes(img, bboxes, labels, title=f"Image {frame_id:06d} from {folder}")