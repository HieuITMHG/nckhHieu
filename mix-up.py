import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

test_img_folder = 'imgs'
img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_img_folder)
images_path = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
images = []

# Load and preprocess images
for path in images_path:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

def mix_up(image1, image2, alpha, target_size=None):
    # If target_size is not provided, use the size of the first image
    if target_size is None:
        target_size = (image1.shape[1], image1.shape[0])  # (width, height)
    
    # Resize image2 to match image1's dimensions
    image2_resized = cv2.resize(image2, target_size, interpolation=cv2.INTER_AREA)
    
    # Generate mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Mix the images
    mixed_img = Image.fromarray(np.uint8(lam * np.array(image1) + (1 - lam) * np.array(image2_resized)))
    return mixed_img

# Check if there are at least two images
if len(images) < 2:
    raise ValueError("At least two images are required for mix-up augmentation.")

# Mix the first two images
mixed_img = mix_up(images[3], images[2], 0.2)

plt.imshow(mixed_img)
plt.axis('off')  # Optional: hide axes
plt.show()