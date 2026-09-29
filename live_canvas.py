import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw

# -------------------------------------------------------------------
# 1. CNN Model Architecture (Phase 1)
# -------------------------------------------------------------------
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerCNN, self).__init__()
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
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# -------------------------------------------------------------------
# 2. Interactive Auto-Transforming Canvas Application
# -------------------------------------------------------------------
class TransformCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer - Smart Canvas")
        self.root.resizable(False, False)

        # Device & Model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitRecognizerCNN().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("digit_recognizer.pth", map_location=self.device))
            self.model.eval()
            print("Loaded trained model weights successfully.")
        except FileNotFoundError:
            print("Error: 'digit_recognizer.pth' not found. Run training script first.")

        # Canvas Dimensions
        self.canvas_size = 300
        
        # PIL Image Buffer for PyTorch inference (Black background, white lines)
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)

        # UI Layout
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(padx=20, pady=10)

        # Status / Confidence Label
        self.result_label = tk.Label(self.root, text="Draw a digit...", font=("Helvetica", 16))
        self.result_label.pack(pady=5)

        # Control Buttons
        self.btn_clear = tk.Button(self.root, text="Clear Canvas", font=("Helvetica", 12), command=self.clear_canvas)
        self.btn_clear.pack(pady=10)

        # Event Bindings
        self.last_x, self.last_y = None, None
        self.has_drawn = False
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Transforms for Model
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def draw(self, event):
        """Draw strokes on mouse drag."""
        self.has_drawn = True
        brush_size = 20  # Bold stroke for clear MNIST downsampling
        
        if self.last_x and self.last_y:
            # Render visible line on Tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=brush_size, capstyle=tk.ROUND, smooth=True, tags="stroke"
            )
            # Render on offscreen PIL buffer
            self.draw_buffer.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=brush_size, joint="round"
            )

        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        """When mouse button is released, run prediction and replace drawing with text."""
        self.last_x, self.last_y = None, None
        
        if self.has_drawn:
            self.transform_drawing_to_text()

    def transform_drawing_to_text(self):
        """Preprocess drawing, predict digit, clear stroke lines, and display font text."""
        # 1. Preprocess image buffer to 28x28 grayscale
        img_resized = self.image_buffer.resize((28, 28), Image.Resampling.BILINEAR)
        tensor_img = self.transform(img_resized).unsqueeze(0).to(self.device)

        # 2. Run prediction
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        pred_digit = str(predicted.item())
        conf_score = confidence.item() * 100

        # 3. Clear existing stroke lines on canvas
        self.canvas.delete("stroke")

        # 4. Draw the predicted number as a crisp digital font at the center
        center_x = self.canvas_size / 2
        center_y = self.canvas_size / 2
        
        self.canvas.create_text(
            center_x, center_y,
            text=pred_digit,
            fill="#00FF66",  # Vibrant neon green font
            font=("Arial", 140, "bold"),
            tags="stroke"
        )

        # 5. Update Status Label
        self.result_label.config(text=f"Predicted: {pred_digit}  ({conf_score:.1f}% confidence)")
        
        # Reset state for next draw
        self.has_drawn = False

    def clear_canvas(self):
        """Clear canvas drawing/text and reset PIL buffer."""
        self.canvas.delete("all")
        self.image_buffer = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_buffer = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Draw a digit...")
        self.has_drawn = False

# -------------------------------------------------------------------
# 3. Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TransformCanvasApp(root)
    root.mainloop()