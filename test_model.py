from PIL import Image
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import create_model as create_model
from create_model import ResNet
import os
from torchvision.datasets import ImageFolder

# Load in model
loaded_model = torch.load('model.pt')

classes = ['compost', 'non_disposable', 'recycle', 'trash'] # assuming that directories/folder names are labels

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# External prediction
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), 'cpu')
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")

if __name__ == "__main__":
    predict_external_image('IMG_3274.jpg') 