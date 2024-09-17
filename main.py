import torch
from torchvision import models
from torchvision.models import VGG19_Weights
import os
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_image, save_image
from nst import run_style_transfer

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained VGG19 model
cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
print("Model loaded.")

# Paths to content and style images
content_image_path = os.path.abspath('D:/Development/neural-style-transfer/data/content/content_image.jpg')
style_image_path = os.path.abspath('D:/Development/neural-style-transfer/data/style/style_image.jpg')
print(f"Content image path: {content_image_path}")
print(f"Style image path: {style_image_path}")
 
# Load content and style images
content_image = load_image(content_image_path)
style_image = load_image(style_image_path, shape=content_image.shape[-2:])
print("Images loaded.")
# Perform style transfer
print("Running style transfer...")
output = run_style_transfer(cnn, content_image, style_image)
print("Style transfer completed.")
# Save the output image
output_image_path = os.path.abspath('D:/Development/neural-style-transfer/data/output_image.jpg')
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
save_image(output, output_image_path)

print(f"Style transfer complete. Output saved at {output_image_path}.")
