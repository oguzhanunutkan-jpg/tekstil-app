import os, base64, io, requests, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

REPLICATE_API = "https://api.replicate.com/v1"

def dataurl_to_bytes(dataurl):
    header, data = dataurl.split(',', 1)
    return base64.b64decode(data)

def resize_image(img_bytes, max_size=768):
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

def to_b64(img_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

@app.route('/')
def index():
    return "DeseniGiydir API calisiyor OK"

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        api_key  = data.get('api_key', '').strip()
        surf_url = data.get('surf_image')
        pat_url  = data.get('pat_image')
        prompt   = data.get('prompt', 'office chair with elegant fabric pattern, realistic product photo')
        strength = float(data.get('strength', 0.7))

        if not api_key:
            return jsonify({'error': 'Replicate API key gerekli'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 768)
        pat_bytes  = resize_image(dataurl_to_bytes(pat_url),  512)

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait"
        }

        # SDXL img2img
        body = {
            "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "input": {
                "image": to_b64(surf_bytes),
                "prompt": prompt + ", realistic fabric texture, professional product photography, high quality",
                "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, watermark",
                "prompt_strength": strength,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }
        }

        res = requests.post(f"{REPLICATE_API}/predictions", json=body, headers=headers, timeout=10)

        if res.status_code == 401:
            return jsonify({'error': 'API key hatalı'}), 401
        if res.status_code == 402:
            return jsonify({'error': 'Replicate krediniz bitti'}), 402

        pred = res.json()
        pred_id = pred.get('id')
        if not pred_id:
            return jsonify({'error': 'Prediction başlatılamadı: ' + str(pred)}), 500

        # Polling — sonuç gelene kadar bekle
        for i in range(60):
            time.sleep(2)
            poll = requests.get(f"{REPLICATE_API}/predictions/{pred_id}", headers=headers, timeout=10)
            p = poll.json()
            status = p.get('status')
            if status == 'succeeded':
                output = p.get('output', [])
                if output:
                    img_url = output[0] if isinstance(output, list) else output
                    img_res = requests.get(img_url, timeout=30)
                    img_b64 = base64.b64encode(img_res.content).decode()
                    return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64})
                return jsonify({'error': 'Çıktı boş'}), 500
            elif status == 'failed':
                return jsonify({'error': 'Model hatası: ' + str(p.get('error', ''))}), 500
            # processing veya starting ise devam et

        return jsonify({'error': 'Zaman aşımı — tekrar deneyin'}), 504

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
