# 🚀 Quick Start Guide

## One-Time Setup (5 minutes)

### 1. Open PowerShell in Project Directory
```powershell
cd "D:\Swapnil\4. Final Year\5. Quantam Project Resarch Papers\Handwritten_Character_Recognition"
```

### 2. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Training (10-20 minutes)

### Full Training (Recommended)
```powershell
python -m src.train
```

### Quick Test (3-5 minutes)
```powershell
python -m src.train --epochs 3 --qnn_epochs 2
```

**Output:** Models saved to `./artifacts/`

---

## Evaluation (1 minute)

```powershell
python -m src.evaluate
```

**Output:** 
- Classification report in terminal
- Confusion matrix → `./artifacts/confusion_matrix.png`

---

## Web UI (Optional)

```powershell
python app.py
```

**Then open:** http://127.0.0.1:5000

---

## Common Commands

### CPU-Only Training
```powershell
python -m src.train --device cpu
```

### GPU Training
```powershell
python -m src.train --device cuda
```

### Low Memory Mode
```powershell
python -m src.train --batch_size 64
```

### Custom Paths
```powershell
python -m src.train --train_dir ./data/my_dataset/Train --test_dir ./data/my_dataset/Test
```

---

## Troubleshooting

### Can't activate venv?
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Out of memory?
```powershell
python -m src.train --batch_size 32 --device cpu
```

### Module not found?
Make sure you're in the project root and use `python -m src.train` (not `python src/train.py`)

---

## File Locations

| File | Location |
|------|----------|
| Best CNN model | `./artifacts/cnn_best.pt` |
| PCA transformer | `./artifacts/pca.joblib` |
| QNN model | `./artifacts/qnn_last.pt` |
| Confusion matrix | `./artifacts/confusion_matrix.png` |

---

## Next Steps

1. ✅ Train the model
2. ✅ Evaluate performance
3. ✅ Test with web UI
4. 📊 Analyze confusion matrix
5. 🔧 Fine-tune hyperparameters
6. 📝 Document results

---

For detailed documentation, see **README.md**
