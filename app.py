import os, base64, io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

def dataurl_to_bytes(dataurl):
    _, data = dataurl.split(',', 1)
    return base64.b64decode(data)

def resize_image(img_bytes, max_size=512):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size/w, max_size/h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()

@app.route('/')
def index():
    return "DeseniGiydir API calisiyor OK"

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data     = request.json
        hf_token = data.get('api_key', '').strip()
        surf_url = data.get('surf_image')
        prompt   = data.get('prompt', 'office chair with elegant fabric pattern, realistic product photo')

        if not hf_token:
            return jsonify({'error': 'Hugging Face token gerekli'}), 400
        if not surf_url:
            return jsonify({'error': 'Görsel eksik'}), 400

        full_prompt = prompt + ", realistic fabric texture, professional studio product photography, high quality, sharp focus, detailed upholstery"

        # fal-ai text-to-image — FLUX modeli
        client = InferenceClient(
            provider="fal-ai",
            api_key=hf_token,
        )

        result = client.text_to_image(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, distorted, deformed, cartoon, watermark, text",
            model="black-forest-labs/FLUX.1-dev",
            width=768,
            height=768,
        )

        if not isinstance(result, Image.Image):
            raise ValueError("Beklenmedik yanıt tipi")

        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + img_b64})

    except Exception as e:
        err = str(e)
        if '401' in err or 'unauthorized' in err.lower():
            return jsonify({'error': 'Token hatalı.'}), 401
        if '402' in err or 'credit' in err.lower() or 'billing' in err.lower():
            return jsonify({'error': 'fal-ai krediniz bitti.'}), 402
        return jsonify({'error': err}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
