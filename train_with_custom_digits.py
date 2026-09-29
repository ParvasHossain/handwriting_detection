import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image, ImageOps

# 1. Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Standard Preprocessing Pipeline
standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3. Augmentation Pipeline for your custom digits
custom_augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4. Custom Dataset Class to process your local PNGs
class CustomDigitsDataset(Dataset):
    def __init__(self, folder_path, multiply_factor=100, transform=None):
        self.transform = transform
        self.data = []
        
        # Supported file extensions
        valid_extensions = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        if not image_files:
            raise FileNotFoundError(f"No images found in {folder_path}. Make sure file names contain the target digit (e.g., 'digit_3.png').")

        # Load images and extract target digit label from filename
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Find digit label in filename (e.g., 'digit_5.png' -> label 5)
            digit_label = None
            for char in filename:
                if char.isdigit():
                    digit_label = int(char)
                    break
            
            if digit_label is not None:
                # Open image and convert to grayscale
                img = Image.open(img_path).convert('L')
                
                # Auto-invert if the background is light (ensure white text on black background)
                # Quick heuristic: if average pixel intensity > 127, invert image
                stat = ImageOps.grayscale(img)
                avg_pixel = sum(stat.getdata()) / len(stat.getdata())
                if avg_pixel > 127:
                    img = ImageOps.invert(img)

                img = img.resize((28, 28))

                # Repeat image samples so custom style gets sufficient weight
                for _ in range(multiply_factor):
                    self.data.append((img.copy(), digit_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 5. Import CNN Model Architecture from Phase 1
from phase1_digits import DigitRecognizerCNN

# 6. Combined Training Script
def train_combined():
    # Load Standard MNIST Dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=standard_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=standard_transform)

    # Load and augment your Custom Dataset (duplicated 100 times per image)
    custom_dataset = CustomDigitsDataset(
        folder_path='./my_digits',
        multiply_factor=100,
        transform=custom_augmentation_transform
    )

    # Combine MNIST + Custom Digits
    combined_train_dataset = ConcatDataset([mnist_dataset, custom_dataset])
    print(f"Loaded {len(mnist_dataset)} MNIST samples + {len(custom_dataset)} Custom samples.")
    print(f"Total training dataset size: {len(combined_train_dataset)}")

    train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize Model & Optimizer
    model = DigitRecognizerCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fine-tuning / Training Loop
    epochs = 4
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate on standard test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Overall Test Accuracy on MNIST: {100 * correct / total:.2f}%")

    # Overwrite weights file for live_canvas.py
    torch.save(model.state_dict(), "digit_recognizer.pth")
    print("Updated model weights successfully saved to digit_recognizer.pth")

if __name__ == "__main__":
    train_combined()