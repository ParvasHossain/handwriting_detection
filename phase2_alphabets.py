import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Custom Transformation to fix EMNIST Orientation
# EMNIST images are rotated 90 degrees CCW and flipped horizontally by default
class FixEMNISTOrientation:
    def __call__(self, img):
        # Rotate 270 deg (or 90 deg clockwise) and transpose
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)

from PIL import Image

transform_train = transforms.Compose([
    FixEMNISTOrientation(),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3256,)) # EMNIST Letters mean and std
])

transform_test = transforms.Compose([
    FixEMNISTOrientation(),
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3256,))
])

# 3. Load EMNIST Letters Dataset
print("Downloading / Loading EMNIST Letters dataset...")
train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform_train)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 4. CNN Architecture (Updated for 26 Classes)
class AlphabetRecognizerCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(AlphabetRecognizerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 32 x 14 x 14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 64 x 7 x 7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes) # 26 output classes (A-Z)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 5. Training & Evaluation
def train(model, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # EMNIST targets are 1-indexed (1=A, 26=Z), adjust to 0-indexed for CrossEntropyLoss
            targets = targets - 1
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            targets = targets - 1
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    print(f"Test Accuracy on EMNIST Letters: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    model = AlphabetRecognizerCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- Training Alphabet Recognizer (Phase 2) ---")
    train(model, criterion, optimizer, epochs=5)
    
    print("--- Evaluating Model ---")
    evaluate(model)
    
    # Save model weights
    torch.save(model.state_dict(), "alphabet_recognizer.pth")
    print("Saved alphabet model weights to 'alphabet_recognizer.pth'")