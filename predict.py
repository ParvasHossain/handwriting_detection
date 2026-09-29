import torch
from torchvision import transforms
from PIL import Image, ImageOps
from phase1_digits import DigitRecognizerCNN

def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load architecture and weights
    model = DigitRecognizerCNN().to(device)
    model.load_state_dict(torch.load("digit_recognizer.pth", map_location=device))
    model.eval()
    
    # Preprocess image to match MNIST expectations (28x28 grayscale, white digit on black background)
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img) # Inverts white background/black text to black background/white text
    img = img.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img)
        prediction = torch.argmax(output, dim=1).item()
        
    print(f"Predicted Digit: {prediction}")

if __name__ == "__main__":
    predict("my_digit.png")