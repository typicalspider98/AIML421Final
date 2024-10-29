# Import necessary libraries
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models  # Import pretrained model module
from torchvision.models import ResNet18_Weights

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = ImageFolder(root='testdata', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the network combining a custom CNN and pretrained ResNet18
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

# Load the trained model
num_classes = 3
model = CombinedModel(num_classes).to(device)
# Load model structure used during training
model.load_state_dict(torch.load('model_fold_2.pth', map_location=device))
model.eval()  # Switch to evaluation mode

# Test function
def test_model(model, test_loader):
    model.eval()  # Ensure the model is in evaluation mode
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for images, labels in test_loader:
            # Use RGB images as input and add grayscale channel
            gray_images = transforms.functional.rgb_to_grayscale(images, num_output_channels=1)
            images = torch.cat((images, gray_images), dim=1)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate accuracy for each class
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    print(f'Overall accuracy on the test data: {100 * correct / total:.2f} %')
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Accuracy of class {test_dataset.classes[i]}: {accuracy:.2f} %')

# Test the model
test_model(model, test_loader)
