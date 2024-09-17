import torch
import os
import torch.optim as optim
from torchvision import models
from utils import load_image, save_image
from losses import ContentLoss, StyleLoss

# Define the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_and_losses(cnn, content_image, style_image, content_layers, style_layers):
    content_losses = []
    style_losses = []
    model = torch.nn.Sequential().to(device)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, torch.nn.ReLU):
            name = f'relu_{i}'
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, content_losses, style_losses

#, style_weight=1e6, content_weight=1
def run_style_transfer(cnn, content_image, style_image, num_steps=20):
    input_image = content_image.clone()
    model, content_losses, style_losses = get_model_and_losses(
        cnn, content_image, style_image, 
        content_layers=['conv_4'], 
        style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    )

    optimizer = optim.LBFGS([input_image.requires_grad_()])

    for step in range(num_steps):
        def closure():
            input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_image)

            content_score = 0
            style_score = 0

            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss

           # loss = content_weight * content_score + style_weight * style_score
           # loss.backward()

            return content_score + style_score

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image
