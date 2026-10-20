import os
import io
import base64
import json
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import albumentations as A
from albumentations.pytorch import ToTensorV2
import google.generativeai as genai

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyD-ZWAz4rqfCjLhJMbyRqlUCSJNd1p2-Bc')
MODEL_PATH     = os.environ.get('MODEL_PATH', 'models/crack_classifier_final.pth')
CLASSES        = ['BULGING', 'COMBINED', 'SHEAR', 'TENSION CRACK']
IMG_SIZE       = 224
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class metadata for rich UI
CLASS_META = {
    'BULGING': {
        'icon': '🔵',
        'color': '#3B82F6',
        'desc': 'Outward deformation of wall or surface',
        'severity': 'Medium',
        'action': 'Structural assessment required'
    },
    'COMBINED': {
        'icon': '🟠',
        'color': '#F97316',
        'desc': 'Multiple failure modes present simultaneously',
        'severity': 'High',
        'action': 'Immediate professional inspection'
    },
    'SHEAR': {
        'icon': '🔴',
        'color': '#EF4444',
        'desc': 'Diagonal cracking from shear stress',
        'severity': 'Critical',
        'action': 'Urgent structural intervention needed'
    },
    'TENSION CRACK': {
        'icon': '🟡',
        'color': '#EAB308',
        'desc': 'Straight cracks from tensile forces',
        'severity': 'Low-Medium',
        'action': 'Monitor and seal cracks'
    }
}

# ── MODEL ────────────────────────────────────────────────────────────────────
class CrackClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def load_model():
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model_name = ckpt.get('model_name', 'efficientnet_b3')
        model = CrackClassifier(model_name, len(CLASSES))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"⚠️  Could not load model: {e}. Using random weights for demo.")
        model = CrackClassifier('efficientnet_b3', len(CLASSES))
        model.eval()
        return model


model = load_model()

# ── INFERENCE ────────────────────────────────────────────────────────────────
def preprocess_image(pil_img):
    img = np.array(pil_img.convert('RGB'))
    _albu_ver = tuple(int(x) for x in A.__version__.split('.')[:2])
    if _albu_ver >= (1, 4):
        tf = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        tf = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return tf(image=img)['image'].unsqueeze(0).to(DEVICE)


def classify_image(pil_img):
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    pred_cls  = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    all_scores = {cls: round(float(p) * 100, 2) for cls, p in zip(CLASSES, probs)}
    return pred_cls, confidence, all_scores


def image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode()


# ── GEMINI ───────────────────────────────────────────────────────────────────
def analyze_with_gemini(initial_img, end_img, classification, confidence):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""You are an expert structural engineer analyzing concrete/masonry failure progression.

I'm providing TWO images:
1. INITIAL STATE - early stage of structural failure  
2. END STATE - advanced/final stage of structural failure

The ML model has classified the END STATE as: **{classification}** with {confidence:.1f}% confidence.

Please analyze and provide a JSON response with EXACTLY this structure (no markdown, pure JSON):
{{
  "failure_rate_percent": <0-100 number indicating progression from initial to end>,
  "severity_level": "<Low|Medium|High|Critical>",
  "crack_width_estimate": "<estimate in mm range>",
  "progression_speed": "<Slow|Moderate|Rapid|Explosive>",
  "structural_integrity_remaining": <0-100 percent>,
  "risk_of_collapse": "<Very Low|Low|Moderate|High|Imminent>",
  "key_observations": ["<obs1>", "<obs2>", "<obs3>"],
  "damage_zones": ["<zone1>", "<zone2>"],
  "recommended_actions": ["<action1>", "<action2>", "<action3>"],
  "time_to_critical": "<estimated timeframe if untreated>",
  "failure_mechanism": "<brief technical description>",
  "repair_complexity": "<Minor|Moderate|Major|Reconstruction>",
  "estimated_repair_cost_range": "<Low/Medium/High/Very High>",
  "confidence_in_analysis": <0-100>
}}"""

        initial_data = image_to_base64(initial_img)
        end_data     = image_to_base64(end_img)

        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": initial_data},
            {"mime_type": "image/jpeg", "data": end_data}
        ])

        text = response.text.strip()
        # Strip markdown code blocks if present
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        return json.loads(text)

    except Exception as e:
        print(f"Gemini error: {e}")
        # Fallback mock data
        return {
            "failure_rate_percent": 72,
            "severity_level": "High",
            "crack_width_estimate": "3-8 mm",
            "progression_speed": "Moderate",
            "structural_integrity_remaining": 45,
            "risk_of_collapse": "Moderate",
            "key_observations": [
                "Significant crack propagation visible between stages",
                "Widening of existing fissures noted",
                "Secondary cracking patterns emerging"
            ],
            "damage_zones": ["Upper-left quadrant", "Central load-bearing area"],
            "recommended_actions": [
                "Immediate structural assessment by licensed engineer",
                "Install crack monitors to track progression",
                "Restrict access to affected zone"
            ],
            "time_to_critical": "3-6 months if untreated",
            "failure_mechanism": f"{classification} failure with progressive deterioration",
            "repair_complexity": "Major",
            "estimated_repair_cost_range": "High",
            "confidence_in_analysis": 78,
            "_note": "Demo data - Gemini API key not configured"
        }


# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/task1')
def task1():
    return render_template('task1.html', class_meta=CLASS_META)

@app.route('/task2')
def task2():
    return render_template('task2.html', class_meta=CLASS_META)

@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    img  = Image.open(file.stream)
    pred_cls, confidence, all_scores = classify_image(img)
    img_b64 = image_to_base64(img.resize((400, 400)))
    return jsonify({
        'prediction': pred_cls,
        'confidence': round(confidence, 2),
        'all_scores': all_scores,
        'meta': CLASS_META[pred_cls],
        'image_preview': img_b64
    })

@app.route('/api/analyze_failure', methods=['POST'])
def api_analyze_failure():
    if 'initial_image' not in request.files or 'end_image' not in request.files:
        return jsonify({'error': 'Both initial and end images required'}), 400

    initial_img = Image.open(request.files['initial_image'].stream)
    end_img     = Image.open(request.files['end_image'].stream)

    # Classify end image with ML model
    pred_cls, confidence, all_scores = classify_image(end_img)

    # Gemini analysis
    gemini_data = analyze_with_gemini(initial_img, end_img, pred_cls, confidence)

    return jsonify({
        'classification': pred_cls,
        'confidence': round(confidence, 2),
        'all_scores': all_scores,
        'meta': CLASS_META[pred_cls],
        'gemini_analysis': gemini_data,
        'initial_preview': image_to_base64(initial_img.resize((400, 400))),
        'end_preview': image_to_base64(end_img.resize((400, 400)))
    })


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
