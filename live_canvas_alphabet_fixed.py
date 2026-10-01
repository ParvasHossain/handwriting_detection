import string
import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

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

INDEX_TO_LETTER = list(string.ascii_uppercase)

class SplitAlphabetCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixed Alphabet Recognizer (A-Z)")
        self.root.resizable(False, False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AlphabetRecognizerCNN(num_classes=26).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("alphabet_recognizer.pth", map_location=self.device))
            self.model.eval()
            print("Loaded updated model weights successfully.")
        except FileNotFoundError:
            print("Error: 'alphabet_recognizer.pth' not found. Run phase2_alphabets_fixed.py first.")

        self.pane_width = 400
        self.pane_height = 400
        self.total_width = self.pane_width * 2

        self.image_buffer = Image.new("L", (self.pane_width, self.pane_height), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)

        self.canvas = tk.Canvas(self.root, width=self.total_width, height=self.pane_height, bg="#121212", highlightthickness=0)
        self.canvas.pack(padx=20, pady=15)

        self.draw_ui_layout()

        self.result_label = tk.Label(self.root, text="Draw a letter (A-Z) on the left side...", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

        self.btn_clear = tk.Button(self.root, text="Clear Canvas", font=("Helvetica", 12, "bold"), bg="#333333", fg="white", command=self.clear_canvas)
        self.btn_clear.pack(pady=10)

        self.last_x, self.last_y = None, None
        self.has_drawn = False
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Standard transformation matching trained inputs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3256,))
        ])

    def draw_ui_layout(self):
        self.canvas.create_line(self.pane_width, 0, self.pane_width, self.pane_height, fill="#00E5FF", width=3)
        self.canvas.create_text(200, 30, text="DRAW LETTER HERE", fill="#888888", font=("Helvetica", 12, "bold"))
        self.canvas.create_text(600, 30, text="REAL LETTER", fill="#888888", font=("Helvetica", 12, "bold"))

    def draw(self, event):
        brush_size = 24  # Slightly thicker stroke for clear center-weighted downsampling
        if event.x < self.pane_width:
            self.has_drawn = True
            if self.last_x is not None and self.last_y is not None:
                if self.last_x < self.pane_width:
                    self.canvas.create_line(
                        self.last_x, self.last_y, event.x, event.y,
                        fill="white", width=brush_size, capstyle=tk.ROUND, smooth=True, tags="drawing_stroke"
                    )
                    self.draw_buffer.line(
                        [self.last_x, self.last_y, event.x, event.y],
                        fill=255, width=brush_size, joint="round"
                    )
            self.last_x, self.last_y = event.x, event.y
        else:
            self.last_x, self.last_y = None, None

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        if self.has_drawn:
            self.predict_and_display()

    def predict_and_display(self):
        # Resize to 28x28 expected by EMNIST
        img_resized = self.image_buffer.resize((28, 28), Image.Resampling.BILINEAR)
        tensor_img = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        pred_idx = predicted.item()
        pred_letter = INDEX_TO_LETTER[pred_idx]
        conf_score = confidence.item() * 100

        self.canvas.delete("prediction_text")
        self.canvas.create_text(
            600, 200,
            text=pred_letter,
            fill="#00E5FF",
            font=("Arial", 180, "bold"),
            tags="prediction_text"
        )

        self.result_label.config(text=f"Predicted Letter: '{pred_letter}'  ({conf_score:.1f}% confidence)")

    def clear_canvas(self):
        self.canvas.delete("drawing_stroke")
        self.canvas.delete("prediction_text")
        self.image_buffer = Image.new("L", (self.pane_width, self.pane_height), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Draw a letter (A-Z) on the left side...")
        self.has_drawn = False

if __name__ == "__main__":
    root = tk.Tk()
    app = SplitAlphabetCanvasApp(root)
    root.mainloop()