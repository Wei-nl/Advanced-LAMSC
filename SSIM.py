import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Define path
raw_images_path = 'data_voc2012/semantic_feature_maps/example'
rec_images_path = 'data_voc2012/rec_images/example'


raw_images = sorted(os.listdir(raw_images_path))
rec_images_TranditionalSC = []
for i in range(3):
    rec_images_TranditionalSC.append(f'rec_img_{i}.jpg')


ssim_values = []


for raw_image_name, rec_image_name in zip(raw_images, rec_images_TranditionalSC):

    aware_image_path = os.path.join(raw_images_path, raw_image_name)
    rec_image_path = os.path.join(rec_images_path, rec_image_name)


    if not os.path.exists(aware_image_path):
        print(f"File not found: {aware_image_path}")
        continue
    if not os.path.exists(rec_image_path):
        print(f"File not found: {rec_image_path}")
        continue

    print(f"Loading aware image: {aware_image_path}")
    print(f"Loading rec image: {rec_image_path}")


    raw_image = cv2.imread(aware_image_path, cv2.IMREAD_GRAYSCALE)
    rec_image = cv2.imread(rec_image_path, cv2.IMREAD_GRAYSCALE)


    if raw_image is None:
        print(f"Error: Unable to load raw image {raw_image_name}")
        continue
    if rec_image is None:
        print(f"Error: Unable to load rec image {rec_image_name}")
        continue
    raw_image = cv2.resize(raw_image, (64, 64), interpolation=cv2.INTER_LINEAR)



    if raw_image.shape != rec_image.shape:
        print(f"Warning: Image sizes do not match for {raw_image_name} and {rec_image_name}")
        continue


    ssim_value, _ = ssim(raw_image, rec_image, full=True)
    print(f"{raw_image_name}SSIM：", ssim_value)

    ssim_values.append(ssim_value)

# Calculate the average of the SSIM
if ssim_values:
    average_ssim = np.mean(ssim_values)
    print(f"Average SSIM of all image pairs: {average_ssim:.4f}")
else:
    print("No valid image pairs to compare.")
