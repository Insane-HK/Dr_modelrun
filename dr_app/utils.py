import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

# Allow safe unpickling of ResNet model (only do this if model source is trusted)
from torch.serialization import add_safe_globals
from torchvision.models.resnet import ResNet, Bottleneck
add_safe_globals([ResNet, Bottleneck])

# Load model from full model .pth file
def load_model(model_path='dr_app/diabetic_retinopathy_full_model.pth'):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# Load it once
model = load_model()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()


# Kirsch edge filter
def apply_kirsch_filter(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or invalid format")

    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # N
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # NE
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # E
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # SE
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # S
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # SW
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # W
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])   # NW
    ]

    max_response = np.zeros_like(image, dtype=np.uint8)
    for kernel in kirsch_kernels:
        filtered = cv2.filter2D(image, -1, kernel)
        max_response = np.maximum(max_response, filtered)

    return max_response
