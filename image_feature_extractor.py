import torch
import lovely_tensors as lt
lt.monkey_patch()
import numpy as np
import tqdm
import cv2
import os
import h5py
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

# Setup command line arguments
parser = argparse.ArgumentParser(description="Extract features using a pretrained image model.")
parser.add_argument('--model_name', type=str, default='vit_large_patch14_dinov2.lvd142m', help='Model name to use for feature extraction.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to image dataset')
args = parser.parse_args()

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_default_device()
print('device:', device)

# Load the pre-trained DINO model
model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move the model to the appropriate device

# Define image transformations
dino_transforms = transforms.Compose([
    transforms.Resize(518),  # Resize images
    transforms.CenterCrop(518),  # Crop the middle 518x518 from the resized image
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalize images
])

# Setup the dataset and DataLoader
dataset_path = args.dataset_path  # Using save_path to find dataset as well
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=dino_transforms)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=dino_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to extract features
def extract_features(data_loader):
    features = []
    labels = []
    with torch.no_grad():  # No need to track gradients
        for inputs, batch_labels in tqdm(data_loader):
            inputs = inputs.to(device)  # Move inputs to the appropriate device
            output = model(inputs)
            features.append(output.cpu())
            labels.append(batch_labels)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

# Extract and save features
train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

# Save the features and labels
os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)  # Ensure the directory exists
torch.save((train_features, train_labels), os.path.join(args.save_path, args.model_name, 'train_features.pt'))
torch.save((test_features, test_labels), os.path.join(args.save_path, args.model_name, 'test_features.pt'))

print("Feature extraction completed and saved.")
