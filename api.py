from PIL import Image
from pathlib import Path
from flask import Flask, jsonify, request
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import create_model as create_model
from create_model import ResNet
import os
from torchvision.datasets import ImageFolder
import test_model
from werkzeug.utils import secure_filename


test = Flask(__name__)


@test.route('/predict', methods=["POST"])
def predict():
   print(request)
   print(request.files)
   file = request.files['file']
   #filename = secure_filename(file.filename)
   typeOfTrash = test_model.predictImage(pathName=file)
   return jsonify({'type': f"{typeOfTrash}"})


if __name__ == "__main__":
   test.run()