import os, base64, io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# ── Helper Functions ──
def dataurl_to_bytes(dataurl):
    """DataURL → bytes"""
    _, data = dataurl.split(',', 1)
    return base64.b64decode(data)

def resize_image(img_bytes, max_size=768):
    """Resize image for model input"""
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ── Routes ──
@app.route('/')
def index():
    return "DeseniGiydir API çalışıyor ✅"

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        hf_token = data.get('api_key', '').strip()
        surf_url = data.get('surf_image')
        pat_url = data.get('pat_image')
        prompt = data.get('prompt', 'office chair with elegant fabric pattern, realistic fabric texture')
        strength = float(data.get('strength', 0.7))

        # Validasyon
        if not hf_token:
            return jsonify({'error': 'Hugging Face token gerekli'}), 400
        if not surf_url:
            return jsonify({'error': 'Ürün görseli eksik'}), 400

        # Görselleri hazırla
        surf_bytes = resize_image(dataurl_to_bytes(surf_url))

        if pat_url:
            prompt += ", upholstery made with the provided fabric pattern"

        # ── Hugging Face Router Client ──
        client = InferenceClient(token=hf_token)  # router endpoint otomatik kullanılıyor

        # ── Image-to-Image ──
        result = client.image_to_image(
            image=surf_bytes,  # bytes input
            model="stabilityai/stable-diffusion-xl-base-1.0",
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, deformed, cartoon, watermark",
            strength=strength
        )

        # PIL Image → Base64
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + img_b64})

    except Exception as e:
        err = str(e)
        if "401" in err.lower():
            return jsonify({'error': 'Token hatalı. HF tokeninizi kontrol edin.'}), 401
        if "credit" in err.lower():
            return jsonify({'error': 'HF krediniz bitti veya model erişimi yok.'}), 402
        return jsonify({'error': err}), 500

# ── Main ──
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
