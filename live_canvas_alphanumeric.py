import string
import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

# 1. Alphanumeric CNN Model
class AlphanumericRecognizerCNN(nn.Module):
    def __init__(self, num_classes=47):
        super(AlphanumericRecognizerCNN, self).__init__()
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

# Character Mapping for EMNIST 'balanced' (0-9, A-Z, select lowercase)
EMNIST_BALANCED_MAPPING = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# 2. Canvas UI
class AlphanumericCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphanumeric Recognizer (0-9 & A-Z)")
        self.root.resizable(False, False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AlphanumericRecognizerCNN(num_classes=47).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("alphanumeric_recognizer.pth", map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            print("Error: 'alphanumeric_recognizer.pth' not found.")

        self.canvas_size = 300
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(padx=20, pady=10)

        self.result_label = tk.Label(self.root, text="Draw a number or letter...", font=("Helvetica", 16))
        self.result_label.pack(pady=5)

        self.btn_clear = tk.Button(self.root, text="Clear Canvas", font=("Helvetica", 12), command=self.clear_canvas)
        self.btn_clear.pack(pady=10)

        self.last_x, self.last_y = None, None
        self.has_drawn = False
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))
        ])

    def draw(self, event):
        self.has_drawn = True
        brush_size = 20
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=brush_size, capstyle=tk.ROUND, smooth=True, tags="stroke"
            )
            self.draw_buffer.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=brush_size, joint="round"
            )
        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        if self.has_drawn:
            self.transform_drawing_to_text()

    def transform_drawing_to_text(self):
        img_resized = self.image_buffer.resize((28, 28), Image.Resampling.BILINEAR)
        tensor_img = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        pred_char = EMNIST_BALANCED_MAPPING[predicted.item()]
        conf_score = confidence.item() * 100

        self.canvas.delete("stroke")
        self.canvas.create_text(
            self.canvas_size / 2, self.canvas_size / 2,
            text=pred_char,
            fill="#00FF88",
            font=("Arial", 140, "bold"),
            tags="stroke"
        )

        self.result_label.config(text=f"Predicted: {pred_char}  ({conf_score:.1f}% confidence)")
        self.has_drawn = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Draw a number or letter...")
        self.has_drawn = False

if __name__ == "__main__":
    root = tk.Tk()
    app = AlphanumericCanvasApp(root)
    root.mainloop()