import os
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

# Define augmentation parameters
augmentations = [
    iaa.Affine(rotate=(-5, 0)),
    iaa.Affine(rotate=(-4, 0)),
    iaa.Affine(rotate=(-3, 0)),
    iaa.Affine(rotate=(-2, 0)),
    iaa.Affine(rotate=(-1, 0)),
    iaa.Affine(rotate=(0, 5)),  
    iaa.Affine(rotate=(0, 4)),
    iaa.Affine(rotate=(0, 3)),
    iaa.Affine(rotate=(0, 2)),
    iaa.Affine(rotate=(0, 1)),
    iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255)),  # Adding Gaussian noise
    iaa.GaussianBlur(sigma=(0.0, 1.0)),  # Blurring
    iaa.Fliplr(1.0),  # Flipping left to right
    iaa.PiecewiseAffine(scale=(0.01, 0.05))  # Distorting
]

# Specify the directory containing your images
input_dir = "C:/Users/dipto/Desktop/gym pose detection/squatties/stand"
output_dir = "C:/Users/dipto/Desktop/gym pose detection/squatties/stand aug"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg", ".webp", ".PNG")):
        image_path = os.path.join(input_dir, filename)
        image = np.array(Image.open(image_path))

        # Apply each augmentation
        for i, aug in enumerate(augmentations):
            augmented_image = aug(image=image)
            
            # Convert RGBA to RGB if necessary
            if augmented_image.shape[-1] == 4:
                augmented_image = augmented_image[..., :3]  # Slice the RGBA channels to extract RGB

            output_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}.jpg"  # Modify the filename for each augmentation
            output_path = os.path.join(output_dir, output_filename)
            
            # Save augmented image
            Image.fromarray(augmented_image).save(output_path)
            print(f"Saved augmented image: {output_path}")

print("Augmentation completed.")
