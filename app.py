import os, base64, io, requests, time, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "DeseniGiydir API calisiyor OK"

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

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 512)

        headers = {
            "Authorization": f"Bearer {hf_token}",
        }

        full_prompt = prompt + ", realistic fabric texture, professional product photography, high quality, sharp focus, detailed"

        # timbrooks/instruct-pix2pix — HF router'da çalışan img2img modeli
        # multipart/form-data olarak gönder
        files = {
            'inputs': ('image.jpg', io.BytesIO(surf_bytes), 'image/jpeg'),
        }
        form_data = {
            'parameters': json.dumps({
                "prompt": full_prompt,
                "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, watermark, text",
                "num_inference_steps": 20,
                "image_guidance_scale": 1.5,
                "guidance_scale": 7.0,
            })
        }

        url = "https://router.huggingface.co/hf-inference/models/timbrooks/instruct-pix2pix"

        res = requests.post(url, headers=headers, files=files, data=form_data, timeout=90)

        # Model yükleniyor
        if res.status_code == 503:
            try:
                wait = res.json().get('estimated_time', 20)
            except:
                wait = 20
            time.sleep(min(float(wait), 30))
            res = requests.post(url, headers=headers, files=files, data=form_data, timeout=120)

        if res.status_code == 401:
            return jsonify({'error': 'HF token hatalı. Write yetkili token oluşturun.'}), 401

        if res.status_code == 200 and 'image' in res.headers.get('content-type', ''):
            img_b64 = base64.b64encode(res.content).decode()
            return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + img_b64})

        # Hata detayı
        try:
            err = res.json()
            msg = err.get('error', str(err))
        except:
            msg = f"HTTP {res.status_code}: {res.text[:200]}"

        return jsonify({'error': msg}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
