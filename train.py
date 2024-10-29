# Import necessary libraries
import os
import shutil
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import models  # Import pretrained model module
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
from torchvision.models import ResNet18_Weights

# Step 1: Data Exploration (EDA)
# Check if image sizes match the expected size (300x300)
data_path = './traindata'  # Dataset path
classes = ['cherry', 'strawberry', 'tomato']  # Class names

def check_image_sizes(data_path, classes, target_size=(300, 300)):
    """
    Check if each image in the dataset has the expected size and count the number of abnormal images.
    """
    normal_count = {cls: 0 for cls in classes}
    abnormal_count = {cls: 0 for cls in classes}

    for cls in classes:
        class_path = os.path.join(data_path, cls)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    if img.size == target_size:
                        normal_count[cls] += 1
                    else:
                        abnormal_count[cls] += 1
                        # Skip images that do not match the target size
            except Exception as e:
                print(f"Unable to read image: {img_path}, Error: {e}")
                abnormal_count[cls] += 1
                # Skip images that cannot be read

    # Output the count of normal and abnormal images
    for cls in classes:
        print(f"Class {cls}: Normal images = {normal_count[cls]}, Abnormal images = {abnormal_count[cls]}")

    return normal_count, abnormal_count

# Call the function to check image sizes and remove non-matching images
normal_count, abnormal_count = check_image_sizes(data_path, classes)

# Step 2: Data Preprocessing and Augmentation
# Use only images with normal sizes to build the dataset
def load_images(data_path, classes, target_size=(300, 300)):
    """
    Load all images with the expected size from the dataset.
    """
    images = []
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    if img.size == target_size:
                        images.append((img.convert('RGB').copy(), cls))
            except Exception as e:
                print(f"Unable to read image: {img_path}, Error: {e}")
    return images

# Get all images with the expected size
all_images = load_images(data_path, classes)

# Define data augmentation and normalization transforms
data_augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # Horizontal flip
    transforms.RandomVerticalFlip(p=1.0),  # Vertical flip
    transforms.RandomChoice(
        [transforms.RandomRotation(90), transforms.RandomRotation(180), transforms.RandomRotation(270)]),
    # Random rotation of 90, 180, 270 degrees
])

# Normalization transform
normalize_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Custom dataset for data augmentation and normalization
class AugmentedDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images) * 8  # Each image will be expanded to 8 images

    def __getitem__(self, idx):
        # Get the original image and label
        original_idx = idx // 8
        img, label = self.images[original_idx]

        # Data augmentation
        img_aug = img.copy()
        if idx % 8 == 1:
            img_aug = data_augmentation_transform(img_aug)  # Augment image
        elif idx % 8 == 2:
            img_aug = transforms.functional.hflip(img_aug)  # Horizontal flip
        elif idx % 8 == 3:
            img_aug = transforms.functional.vflip(img_aug)  # Vertical flip
        elif idx % 8 == 4:
            img_aug = transforms.functional.rotate(img_aug, 90)  # Rotate 90 degrees
        elif idx % 8 == 5:
            img_aug = transforms.functional.rotate(img_aug, 180)  # Rotate 180 degrees
        elif idx % 8 == 6:
            img_aug = transforms.functional.rotate(img_aug, 270)  # Rotate 270 degrees
        elif idx % 8 == 7:
            img_aug = transforms.functional.vflip(transforms.functional.rotate(img_aug, 90))  # Rotate 90 and vertical flip

        # Convert the image to grayscale and combine with RGB image
        gray_img = transforms.functional.to_grayscale(img_aug, num_output_channels=1)  # Convert to grayscale
        gray_img = transforms.ToTensor()(gray_img)  # Convert to Tensor
        img_processed = normalize_transform(img_aug)  # Process RGB image
        img_processed = torch.cat((img_processed, gray_img), dim=0)  # Concatenate RGB and grayscale images

        return img_processed, classes.index(label)

# Build the augmented dataset
augmented_dataset = AugmentedDataset(all_images, transform=data_augmentation_transform)

# Cross-validation setup
kf = KFold(n_splits=2, shuffle=True, random_state=42)
accuracies = []

# Step 3: Build the network combining custom CNN and pretrained model
class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        # Custom CNN part
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.3)

        # Pretrained ResNet18 model
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer of ResNet

        # Fully connected layer to combine custom CNN and ResNet features
        self.fc1 = nn.Linear(64 * 37 * 37 + 512, 256)  # ResNet output dimension is 512, custom CNN output is 64 * 37 * 37
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Custom CNN feature extraction
        x_cnn = self.pool(F.leaky_relu(self.conv1(x)))
        x_cnn = self.pool(F.relu(self.conv2(x_cnn)))
        x_cnn = self.pool(F.relu(self.conv3(x_cnn)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)  # Flatten the tensor

        # Reduce output channels to 3 (RGB) to fit the input of the pretrained model
        x_resnet_input = x[:, :3, :, :]
        x_resnet = self.resnet(x_resnet_input)  # Feature extraction through pretrained ResNet

        # Concatenate features from custom CNN and ResNet
        x_combined = torch.cat((x_cnn, x_resnet), dim=1)

        # Classification through fully connected layers
        x = self.dropout(F.relu(self.fc1(x_combined)))
        x = self.fc2(x)
        return x

# Hyperparameter settings
num_classes = 3  # Number of classes
learning_rate = 0.001
num_epochs = 50

# Train the model using cross-validation
# Split the training, validation, and test sets with a 9:1 ratio for training and validation
for fold, (train_idx, test_idx) in enumerate(kf.split(augmented_dataset)):
    print(f'Fold {fold + 1}/{kf.n_splits}')
    val_size = int(0.1 * len(train_idx))
    train_idx, val_idx = train_idx[val_size:], train_idx[:val_size]
    train_subset = Subset(augmented_dataset, train_idx)
    val_subset = Subset(augmented_dataset, val_idx)
    test_subset = Subset(augmented_dataset, test_idx)

    batchNum = 64
    train_loader = DataLoader(train_subset, batch_size=batchNum, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batchNum, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batchNum, shuffle=False)

    # Model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(num_classes).to(device)
    class_counts = [normal_count['cherry'], normal_count['strawberry'], normal_count['tomato']]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Train the model
    # Implement early stopping
    early_stopping_patience = 10
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Evaluate on validation set
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

    # Test the model
    model.eval()
    correct = 0

    # Save the trained model
    torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')
