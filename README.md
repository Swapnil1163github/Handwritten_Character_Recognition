# Handwritten Devanagari Character Recognition
## CNN + PCA + Quantum Neural Network (Hybrid Model)

A quantum-classical hybrid deep learning system for recognizing handwritten Devanagari characters and digits using PyTorch, PennyLane, and scikit-learn.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Structure](#dataset-structure)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Running the Web UI](#running-the-web-ui)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a **two-stage hybrid quantum-classical model**:

### Stage 1: CNN Feature Extraction
- Convolutional Neural Network (PyTorch) extracts deep features from 32×32 grayscale images
- 3 conv blocks (32→64→128 filters) with BatchNorm, ReLU, MaxPool
- Fully connected classifier with dropout for regularization
- Trained on 46 classes (36 Devanagari characters + 10 digits)

### Stage 2: PCA + Quantum Neural Network
- PCA reduces CNN features from 2048 → 8 dimensions
- Variational Quantum Circuit (PennyLane) with:
  - Angle embedding (Y-rotation gates)
  - Strongly entangling layers
  - Expectation value measurements
- Fine-tuned for quantum-enhanced classification

---

## ✅ Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows (instructions provided for PowerShell)
- **Hardware**: 
  - CPU: Any modern processor (QNN runs on CPU)
  - GPU: NVIDIA GPU with CUDA (optional, for faster CNN training)
  - RAM: 8GB minimum, 16GB recommended
- **Dataset**: Devanagari handwritten character images organized in folders

---

## 🚀 Installation

### Step 1: Navigate to Project Directory
```powershell
cd "D:\Swapnil\4. Final Year\5. Quantam Project Resarch Papers\Handwritten_Character_Recognition"
```

### Step 2: Create Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

**If you get an execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies
```powershell
pip install -r requirements.txt
```

**Installed packages:**
- `torch`, `torchvision` - Deep learning framework
- `pennylane` - Quantum machine learning
- `scikit-learn` - PCA and metrics
- `matplotlib`, `seaborn` - Visualization
- `flask` - Web UI (optional)
- `tqdm`, `joblib`, `pandas` - Utilities

---

## 📁 Dataset Structure

Ensure your dataset follows this structure:

```
Handwritten_Character_Recognition/
├── data/
│   └── devanagari_dataset/
│       ├── Train/
│       │   ├── character_1_ka/      (~1700 images)
│       │   ├── character_2_kha/
│       │   ├── ...
│       │   ├── character_36_gya/
│       │   ├── digit_0/
│       │   ├── digit_1/
│       │   ├── ...
│       │   └── digit_9/
│       └── Test/
│           ├── character_1_ka/      (~300 images)
│           ├── character_2_kha/
│           ├── ...
│           └── digit_9/
```

**Requirements:**
- Each subfolder name becomes the class label
- Images can be any format (PNG, JPG, etc.)
- Images will be auto-converted to grayscale and resized to 32×32

---

## 🏋️ Training the Model

### Basic Training (Recommended)
```powershell
python -m src.train --epochs 10 --batch_size 128 --pca_components 8 --qnn_layers 2 --qnn_epochs 5
```

**What happens:**
1. **Stage 1 (CNN Training):**
   - Trains for 10 epochs
   - Saves best model to `./artifacts/cnn_best.pt`
   - Saves final model to `./artifacts/cnn_last.pt`

2. **Stage 2 (PCA + QNN):**
   - Freezes CNN weights
   - Extracts features from all training/test images
   - Applies PCA dimensionality reduction
   - Trains quantum classifier for 5 epochs
   - Saves PCA to `./artifacts/pca.joblib`
   - Saves QNN to `./artifacts/qnn_last.pt`

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_dir` | `./data/devanagari_dataset/Train` | Path to training data |
| `--test_dir` | `./data/devanagari_dataset/Test` | Path to test data |
| `--epochs` | `10` | CNN training epochs |
| `--batch_size` | `128` | Batch size for training |
| `--lr` | `0.001` | Learning rate for CNN |
| `--dropout` | `0.5` | Dropout rate |
| `--device` | `auto` | `cuda` or `cpu` |
| `--pca_components` | `8` | PCA dimensions (= num qubits) |
| `--qnn_layers` | `2` | Quantum circuit depth |
| `--qnn_epochs` | `5` | QNN training epochs |
| `--qnn_lr` | `0.001` | Learning rate for QNN |
| `--save_dir` | `./artifacts` | Output directory |

### Example Training Outputs

```
Stage 1: Training CNN on 46 classes for 10 epochs
Epoch 01: train_loss=2.3456 train_acc=0.3421 | test_loss=2.1234 test_acc=0.4123
Epoch 02: train_loss=1.7654 train_acc=0.5234 | test_loss=1.5432 test_acc=0.5876
...
Epoch 10: train_loss=0.4321 train_acc=0.8765 | test_loss=0.5234 test_acc=0.8456

Stage 2: PCA + QNN
QNN Epoch 01: train_loss=1.2345 train_acc=0.6543 | test_loss=1.1234 test_acc=0.6789
...
QNN Epoch 05: train_loss=0.8765 train_acc=0.7654 | test_loss=0.9012 test_acc=0.7543
```

### Advanced Training Options

**Quick test run (fewer epochs):**
```powershell
python -m src.train --epochs 3 --qnn_epochs 2
```

**CPU-only training:**
```powershell
python -m src.train --device cpu
```

**GPU training (if CUDA available):**
```powershell
python -m src.train --device cuda
```

**Memory-constrained systems:**
```powershell
python -m src.train --batch_size 64
```

**Smaller quantum circuit:**
```powershell
python -m src.train --pca_components 4 --qnn_layers 1
```

---

## 📊 Evaluation

### Generate Confusion Matrix and Metrics
```powershell
python -m src.evaluate --weights ./artifacts/cnn_best.pt --save_cm ./artifacts/confusion_matrix.png
```

**Outputs:**
1. **Terminal:** Classification report with precision, recall, F1-score per class
2. **File:** Confusion matrix heatmap saved to `./artifacts/confusion_matrix.png`

### Example Evaluation Output
```
                    precision    recall  f1-score   support

    character_1_ka       0.92      0.89      0.90       300
   character_2_kha       0.88      0.91      0.89       300
   character_3_ga        0.85      0.87      0.86       300
   ...
           digit_0       0.95      0.93      0.94       300
           digit_1       0.91      0.94      0.92       300
   ...

          accuracy                           0.87     13800
         macro avg       0.87      0.87      0.87     13800
      weighted avg       0.87      0.87      0.87     13800
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_dir` | `./data/devanagari_dataset/Train` | For class mapping |
| `--test_dir` | `./data/devanagari_dataset/Test` | Test images |
| `--weights` | `./artifacts/cnn_best.pt` | Model weights |
| `--save_cm` | `./artifacts/confusion_matrix.png` | Output path |
| `--batch_size` | `128` | Batch size |
| `--device` | `auto` | `cuda` or `cpu` |

---

## 🌐 Running the Web UI

### Step 1: Ensure Model is Trained
Make sure `./artifacts/cnn_best.pt` exists (run training first).

### Step 2: Start Flask Server
```powershell
python app.py
```

### Step 3: Open Browser
Navigate to: **http://127.0.0.1:5000**

### Step 4: Upload and Predict
1. Click "Choose File" and select a handwritten character image
2. Click "Predict Character"
3. View prediction with confidence score

**Example output:**
```
Prediction: character_5_cha (confidence: 94.23%)
```

### UI Features
- Accepts any image format (PNG, JPG, etc.)
- Auto-converts to grayscale and resizes to 32×32
- Shows predicted class name and confidence percentage
- Modern, responsive design

---

## 📂 Project Structure

```
Handwritten_Character_Recognition/
├── data/
│   └── devanagari_dataset/
│       ├── Train/           # Training images
│       └── Test/            # Test images
├── src/
│   ├── __init__.py          # Package marker
│   ├── preprocess.py        # Data loading & transforms
│   ├── cnn_model.py         # CNN architecture
│   ├── pca_qnn_model.py     # PCA + Quantum NN
│   ├── train.py             # Training orchestration
│   └── evaluate.py          # Evaluation & metrics
├── artifacts/               # Auto-created during training
│   ├── cnn_best.pt          # Best CNN weights
│   ├── cnn_last.pt          # Final CNN weights
│   ├── pca.joblib           # Fitted PCA transformer
│   ├── qnn_last.pt          # QNN weights
│   └── confusion_matrix.png # Evaluation plot
├── app.py                   # Flask web UI
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔧 Troubleshooting

### Issue: "Execution policy error" when activating venv
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `--batch_size 64` or `--batch_size 32`
2. Use CPU: `--device cpu`
3. Close other GPU applications

### Issue: "No module named 'src'"
**Solution:**
Make sure you're running from the project root:
```powershell
cd "D:\Swapnil\4. Final Year\5. Quantam Project Resarch Papers\Handwritten_Character_Recognition"
python -m src.train
```

### Issue: Slow QNN training
**Expected behavior:** QNN runs on CPU and is slower than CNN. 
**Solutions:**
1. Reduce `--pca_components` (fewer qubits)
2. Reduce `--qnn_layers` (shallower circuit)
3. Reduce `--qnn_epochs`

### Issue: "FileNotFoundError: [Errno 2] No such file or directory"
**Solution:**
Verify dataset paths:
```powershell
ls ./data/devanagari_dataset/Train
ls ./data/devanagari_dataset/Test
```

### Issue: Low accuracy
**Solutions:**
1. Train for more epochs: `--epochs 20`
2. Adjust learning rate: `--lr 0.0005`
3. Try different dropout: `--dropout 0.3`
4. Ensure dataset quality (balanced classes, clear images)

### Issue: Flask UI shows wrong predictions
**Solution:**
Ensure you're using the best weights:
```powershell
python app.py  # Automatically loads ./artifacts/cnn_best.pt
```

---

## 🎓 Model Architecture Details

### CNN (Stage 1)
```
Input: 1×32×32 grayscale image
├── Conv2d(1→32, k=3) + BatchNorm + ReLU + MaxPool → 32×16×16
├── Conv2d(32→64, k=3) + BatchNorm + ReLU + MaxPool → 64×8×8
├── Conv2d(64→128, k=3) + BatchNorm + ReLU + MaxPool → 128×4×4
├── Flatten → 2048
├── Linear(2048→256) + ReLU + Dropout(0.5)
└── Linear(256→46) → Logits
```

### PCA + QNN (Stage 2)
```
CNN Features (2048-dim)
├── PCA → 8-dim
├── Quantum Circuit:
│   ├── AngleEmbedding (Y-rotations)
│   ├── StronglyEntanglingLayers (2 layers)
│   └── PauliZ measurements → 8 expectation values
└── Linear(8→46) → Logits
```

---

## 📈 Expected Performance

| Metric | CNN (Stage 1) | PCA+QNN (Stage 2) |
|--------|---------------|-------------------|
| Training Time | ~10-15 min (GPU) | ~5-10 min (CPU) |
| Test Accuracy | 85-90% | 75-80% |
| Inference Speed | ~100 img/sec | ~10 img/sec |

**Note:** QNN performance depends on circuit depth and number of qubits.

---

## 🔮 Future Enhancements

- [ ] Add support for English/Roman character datasets
- [ ] Implement quantum-aware training (end-to-end)
- [ ] Add data augmentation (rotation, scaling, noise)
- [ ] Deploy to cloud (AWS/Azure)
- [ ] Real-time webcam character recognition
- [ ] Model quantization for mobile deployment

---

## 📝 Citation

If you use this project in your research, please cite:

```
@misc{devanagari_qnn_2025,
  title={Handwritten Devanagari Character Recognition using Quantum-Classical Hybrid Neural Networks},
  author={Your Name},
  year={2025},
  institution={Your University}
}
```

---

## 📧 Contact

For questions or issues, please contact:
- **Name:** Swapnil
- **Project:** Final Year Quantum Research
- **Institution:** [Your University]

---

## 📜 License

This project is for academic research purposes.

---

**Happy Training! 🚀**
