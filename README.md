# 🖊️ Real-Time Handwritten Character Recognizer

An interactive Deep Learning application built with **PyTorch** and **Tkinter**. It captures handwritten input live on a desktop canvas, classifies digits and letters using Convolutional Neural Networks (CNNs), and auto-transforms your live drawing into clean digital text.

---

## 🌟 Project Roadmap & Features

This project is structured into 4 evolutionary phases:

- [x] **Phase 1: Digit Recognition (0–9)**
  - Trained on the standard MNIST dataset (~99% accuracy).
  - Fine-tuned with custom handwriting PNG samples using PyTorch data augmentation and `ConcatDataset`.
  - Interactive Tkinter canvas that auto-transforms hand-drawn strokes into digital numbers on mouse release.
- [x] **Phase 2: Unified Alphanumeric Recognition (0–9 & A–Z)**
  - Trained on the **EMNIST (Balanced)** dataset across 47 classes.
  - Automatically handles EMNIST matrix rotations/flips.
  - Real-time prediction and digital rendering of numbers, uppercase, and select lowercase characters.
- [ ] **Phase 3: Word Recognition** *(In Progress)*
  - Transitioning to CRNN (CNN + LSTM) architecture with Connectionist Temporal Classification (CTC) loss.
- [ ] **Phase 4: Full Sentence & Line Recognition** *(Planned)*
  - Vision Transformer / TrOCR model integration for full handwritten lines and spacing context.

---

## 📂 Project Structure

```text
handwriting-recognizer/
├── data/                            # Downloaded MNIST / EMNIST datasets
├── my_digits/                       # Custom PNG samples (e.g., digit_0.png)
│   ├── digit_0.png
│   └── digit_1.png
├── phase1_digits.py                 # Phase 1: Train MNIST Digits model
├── phase2_alphanumeric.py           # Phase 2: Train EMNIST Balanced model
├── train_with_custom_digits.py      # Fine-tune MNIST with local custom handwriting
├── live_canvas.py                   # Live canvas for Digits
├── live_canvas_alphanumeric.py      # Live canvas for Alphanumeric (0-9, A-Z)
├── digit_recognizer.pth             # Phase 1 model weights
├── alphanumeric_recognizer.pth      # Phase 2 model weights
├── requirements.txt
└── README.md
