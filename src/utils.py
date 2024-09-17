import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path)

    # Resize the image if it's too large
    if max_size:
        size = max(max_size, max(image.size))
        image = transforms.Resize(size)(image)
    
    if shape is not None:
        image = transforms.Resize(shape)(image)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension and move to device
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor, output_path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(output_path)
