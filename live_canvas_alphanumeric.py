import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

# -------------------------------------------------------------------
# 1. Alphanumeric CNN Architecture
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 2. Dual-Pane Split Canvas Application
# -------------------------------------------------------------------
class SplitCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphanumeric Recognizer - Split View")
        self.root.resizable(False, False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AlphanumericRecognizerCNN(num_classes=47).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("alphanumeric_recognizer.pth", map_location=self.device))
            self.model.eval()
            print("Loaded model weights successfully.")
        except FileNotFoundError:
            print("Error: 'alphanumeric_recognizer.pth' not found. Run phase2_alphanumeric.py first.")

        # Canvas Dimensions
        self.pane_width = 400
        self.pane_height = 400
        self.total_width = self.pane_width * 2  # 800px total width

        # Offscreen PIL Buffer (Only tracks the 400x400 drawing region on the left)
        self.image_buffer = Image.new("L", (self.pane_width, self.pane_height), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)

        # Main Canvas (800x400)
        self.canvas = tk.Canvas(self.root, width=self.total_width, height=self.pane_height, bg="#121212", highlightthickness=0)
        self.canvas.pack(padx=20, pady=15)

        # Draw UI Dividers & Section Headings
        self.draw_ui_layout()

        # Status & Confidence Bar
        self.result_label = tk.Label(self.root, text="Draw a digit or letter on the left side...", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

        # Clear Button
        self.btn_clear = tk.Button(self.root, text="Clear Canvas", font=("Helvetica", 12, "bold"), bg="#333333", fg="white", command=self.clear_canvas)
        self.btn_clear.pack(pady=10)

        # Mouse Event Tracking
        self.last_x, self.last_y = None, None
        self.has_drawn = False
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Transformation pipeline matching EMNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))
        ])

    def draw_ui_layout(self):
        """Draw background grid, split line, and section titles."""
        # Vertical Separator Line in the Middle (x = 400)
        self.canvas.create_line(self.pane_width, 0, self.pane_width, self.pane_height, fill="#00FF88", width=3)

        # Section Labels
        self.canvas.create_text(200, 30, text="DRAW HERE", fill="#888888", font=("Helvetica", 12, "bold"))
        self.canvas.create_text(600, 30, text="REAL CHARACTER", fill="#888888", font=("Helvetica", 12, "bold"))

    def draw(self, event):
        """Allow drawing ONLY within the left half (x < 400)."""
        brush_size = 22  # Thicker brush stroke for comfortable large drawing

        # Restrict mouse input to the left drawing panel
        if event.x < self.pane_width:
            self.has_drawn = True
            
            if self.last_x is not None and self.last_y is not None:
                # Ensure previous point was also on the left pane
                if self.last_x < self.pane_width:
                    # Render visible line on Tkinter canvas
                    self.canvas.create_line(
                        self.last_x, self.last_y, event.x, event.y,
                        fill="white", width=brush_size, capstyle=tk.ROUND, smooth=True, tags="drawing_stroke"
                    )
                    # Render on PIL buffer
                    self.draw_buffer.line(
                        [self.last_x, self.last_y, event.x, event.y],
                        fill=255, width=brush_size, joint="round"
                    )

            self.last_x, self.last_y = event.x, event.y
        else:
            self.last_x, self.last_y = None, None

    def on_release(self, event):
        """On mouse release, run inference and project predicted digit to the right pane."""
        self.last_x, self.last_y = None, None
        if self.has_drawn:
            self.predict_and_display()

    def predict_and_display(self):
        # 1. Resize left panel drawing buffer down to 28x28 expected by EMNIST
        img_resized = self.image_buffer.resize((28, 28), Image.Resampling.BILINEAR)
        tensor_img = self.transform(img_resized).unsqueeze(0).to(self.device)

        # 2. Run model prediction
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        pred_char = EMNIST_BALANCED_MAPPING[predicted.item()]
        conf_score = confidence.item() * 100

        # 3. Clear previous prediction from the right panel (x = 400..800)
        self.canvas.delete("prediction_text")

        # 4. Render predicted digital character at the center of the right panel (x = 600, y = 200)
        self.canvas.create_text(
            600, 200,
            text=pred_char,
            fill="#00FF88",  # Neon Green Digital Font
            font=("Arial", 180, "bold"),
            tags="prediction_text"
        )

        # 5. Update Status Bar
        self.result_label.config(text=f"Predicted Character: '{pred_char}'  ({conf_score:.1f}% confidence)")

    def clear_canvas(self):
        """Clear both drawing strokes and prediction text."""
        self.canvas.delete("drawing_stroke")
        self.canvas.delete("prediction_text")
        self.image_buffer = Image.new("L", (self.pane_width, self.pane_height), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Draw a digit or letter on the left side...")
        self.has_drawn = False

# -------------------------------------------------------------------
# 3. Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SplitCanvasApp(root)
    root.mainloop()