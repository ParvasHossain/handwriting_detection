import string
import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

# -------------------------------------------------------------------
# 1. Alphabet CNN Architecture
# -------------------------------------------------------------------
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

# Index-to-Letter Mapping (0 -> A, 1 -> B, ..., 25 -> Z)
INDEX_TO_LETTER = list(string.ascii_uppercase)

# -------------------------------------------------------------------
# 2. Interactive Alphabet Canvas Application
# -------------------------------------------------------------------
class AlphabetCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Alphabet Recognizer (A-Z)")
        self.root.resizable(False, False)

        # Device & Model Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AlphabetRecognizerCNN(num_classes=26).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("alphabet_recognizer.pth", map_location=self.device))
            self.model.eval()
            print("Loaded alphabet recognizer weights successfully.")
        except FileNotFoundError:
            print("Error: 'alphabet_recognizer.pth' not found. Run phase2_alphabets.py first.")

        # Canvas Setup
        self.canvas_size = 300
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)

        # UI Layout
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(padx=20, pady=10)

        self.result_label = tk.Label(self.root, text="Draw a letter (A-Z)...", font=("Helvetica", 16))
        self.result_label.pack(pady=5)

        self.btn_clear = tk.Button(self.root, text="Clear Canvas", font=("Helvetica", 12), command=self.clear_canvas)
        self.btn_clear.pack(pady=10)

        # Event Bindings
        self.last_x, self.last_y = None, None
        self.has_drawn = False
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Normalization Transform (matching EMNIST parameters)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3256,))
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
        # Resize drawing buffer to 28x28 expected by EMNIST
        img_resized = self.image_buffer.resize((28, 28), Image.Resampling.BILINEAR)
        tensor_img = self.transform(img_resized).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        pred_idx = predicted.item()
        pred_letter = INDEX_TO_LETTER[pred_idx]
        conf_score = confidence.item() * 100

        # Replace strokes with clean text font
        self.canvas.delete("stroke")
        self.canvas.create_text(
            self.canvas_size / 2, self.canvas_size / 2,
            text=pred_letter,
            fill="#00E5FF",  # Cyan digital font
            font=("Arial", 140, "bold"),
            tags="stroke"
        )

        self.result_label.config(text=f"Predicted Letter: {pred_letter}  ({conf_score:.1f}% confidence)")
        self.has_drawn = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Draw a letter (A-Z)...")
        self.has_drawn = False

if __name__ == "__main__":
    root = tk.Tk()
    app = AlphabetCanvasApp(root)
    root.mainloop()