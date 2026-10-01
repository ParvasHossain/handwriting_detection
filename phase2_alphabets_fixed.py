import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix EMNIST orientation issue (EMNIST raw images are transposed and flipped)
def fix_emnist_orientation(tensor):
    return torch.transpose(tensor, 1, 2).flip(1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(fix_emnist_orientation),
    transforms.Normalize((0.1736,), (0.3256,))
])

print("Downloading/Loading EMNIST Letters dataset...")
train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

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

def train(model, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            targets = targets - 1  # Convert 1-26 to 0-25
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    model = AlphabetRecognizerCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("--- Training Corrected Alphabet Recognizer ---")
    train(model, criterion, optimizer, epochs=5)
    
    torch.save(model.state_dict(), "alphabet_recognizer.pth")
    print("Saved re-aligned model weights to 'alphabet_recognizer.pth'")