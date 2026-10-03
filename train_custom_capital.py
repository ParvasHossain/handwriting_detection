import os
import glob
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image, ImageOps

# 1. Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# EMNIST Correction Transformation
class FixEMNISTOrientation:
    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)

# Transformations
standard_transform = transforms.Compose([
    FixEMNISTOrientation(),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3256,))
])

# Augmentation for your custom alphabet PNGs
custom_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3256,))
])

# 2. Custom Dataset Class for 'capital' folder
class CustomCapitalDataset(Dataset):
    def __init__(self, folder_path, multiply_factor=200, transform=None):
        self.transform = transform
        self.data = []
        
        # Build mapping from letter to class index (0 for 'A', 1 for 'B', ..., 25 for 'Z')
        letter_to_index = {letter: idx for idx, letter in enumerate(string.ascii_uppercase)}

        # Find all image files in capital/
        valid_exts = ("*.png", "*.jpg", "*.jpeg")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_paths:
            raise FileNotFoundError(f"No images found in folder '{folder_path}'. Ensure files match 'Cap_A.png' format.")

        for img_path in image_paths:
            filename = os.path.basename(img_path) # e.g., 'Cap_A.png'
            
            # Extract letter character from filename (Cap_A.png -> 'A')
            letter = None
            for char in filename:
                if char.isupper() and char in letter_to_index:
                    letter = char
                    break
            
            if letter is not None:
                class_idx = letter_to_index[letter]
                img = Image.open(img_path).convert('L')
                
                # Auto-invert if image has light background (ensure white text on black background)
                stat = ImageOps.grayscale(img)
                avg_pixel = sum(stat.getdata()) / len(stat.getdata())
                if avg_pixel > 127:
                    img = ImageOps.invert(img)

                img = img.resize((28, 28))

                # Duplicate custom style (e.g. 200x) so model prioritizes your style
                for _ in range(multiply_factor):
                    self.data.append((img.copy(), class_idx))
            else:
                print(f"Skipping {filename}: Could not determine target capital letter (A-Z).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 3. CNN Architecture (26 Output Classes)
class AlphabetRecognizerCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(AlphabetRecognizerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 4. Training Function
def train_with_custom_capital():
    print("Loading standard EMNIST Letters dataset...")
    emnist_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=standard_transform)
    
    # EMNIST targets are 1-indexed (1=A, 26=Z). Convert EMNIST targets to 0-indexed (0=A, 25=Z)
    emnist_dataset.targets = emnist_dataset.targets - 1

    print("Loading custom capital dataset from './capital'...")
    custom_dataset = CustomCapitalDataset(
        folder_path='./capital',
        multiply_factor=200,  # Boost custom style weight
        transform=custom_augmentation
    )

    combined_dataset = ConcatDataset([emnist_dataset, custom_dataset])
    print(f"Loaded {len(emnist_dataset)} EMNIST letters + {len(custom_dataset)} Custom capital samples.")
    print(f"Total training dataset size: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    model = AlphabetRecognizerCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    epochs = 5
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

    # Overwrite alphabet recognizer weights
    torch.save(model.state_dict(), "alphabet_recognizer.pth")
    print("Model weights successfully saved to 'alphabet_recognizer.pth'!")

if __name__ == "__main__":
    train_with_custom_capital()