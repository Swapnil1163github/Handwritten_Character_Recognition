from flask import Flask, request, render_template_string
import torch
from torchvision import transforms
from PIL import Image
import io
import os

from src.cnn_model import CNNClassifier
from src.preprocess import get_dataloaders

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Devanagari Character Recognition</title>
<style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
    h1 { color: #333; }
    .upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
    input[type=file] { margin: 10px 0; }
    input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    input[type=submit]:hover { background: #45a049; }
    .result { background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; font-size: 18px; }
</style>
<h1>🔤 Devanagari Character Recognition</h1>
<p>Upload a handwritten Devanagari character or digit image (grayscale or color, will be resized to 32×32)</p>
<div class="upload-form">
    <form method=post enctype=multipart/form-data>
      <input type=file name=file accept="image/*" required>
      <input type=submit value="Predict Character">
    </form>
</div>
{% if pred_text %}
<div class="result">
    <strong>Prediction:</strong> {{ pred_text }}
</div>
{% endif %}
"""

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# Load trained model and class mapping
WEIGHTS_PATH = "./artifacts/cnn_best.pt"
TRAIN_DIR = "./data/devanagari_dataset/Train"

# Get class mapping
_, _, class_to_idx, idx_to_class = get_dataloaders(train_dir=TRAIN_DIR, batch_size=1)
num_classes = len(class_to_idx)

# Load model
model = CNNClassifier(num_classes=num_classes)
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    print(f"✓ Loaded weights from {WEIGHTS_PATH}")
else:
    print(f"⚠ Warning: {WEIGHTS_PATH} not found. Using untrained model.")
model.eval()

@app.route('/', methods=['GET', 'POST'])
def upload():
    pred_text = ''
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = Image.open(io.BytesIO(file.read())).convert('L')
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                pred_idx = logits.argmax(dim=1).item()
                pred_class = idx_to_class[pred_idx]
                confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()
            pred_text = f"{pred_class} (confidence: {confidence:.2%})"
    return render_template_string(HTML, pred_text=pred_text)

if __name__ == '__main__':
    app.run(debug=True)
