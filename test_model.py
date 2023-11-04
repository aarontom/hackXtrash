from PIL import Image
from pathlib import Path

# Load in model
loaded_model = torch.load('load/from/path/model.pt') # Add later

# External prediction
def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")

predict_external_image('someimage.jpg') # Add later