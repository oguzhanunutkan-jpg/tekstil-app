import os, base64, io, requests, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

HF_API = "https://router.huggingface.co/hf-inference/models"

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
        strength = float(data.get('strength', 0.6))
        steps    = int(data.get('steps', 25))

        if not hf_token:
            return jsonify({'error': 'Hugging Face token gerekli'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 768)

        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
            "X-Wait-For-Model": "true"
        }

        # HF img2img — stable-diffusion-img2img pipeline
        payload = {
            "inputs": prompt + ", realistic fabric texture, professional product photography, high quality, sharp focus",
            "parameters": {
                "image": base64.b64encode(surf_bytes).decode('utf-8'),
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, watermark, text"
            }
        }

        # Önce img2img modeli dene
        models = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
        ]

        result_bytes = None
        last_error = ""

        for model in models:
            try:
                res = requests.post(
                    f"{HF_API}/{model}",
                    json=payload,
                    headers=headers,
                    timeout=60
                )

                if res.status_code == 503:
                    # Model yükleniyor, bekle
                    wait = res.json().get('estimated_time', 20)
                    time.sleep(min(wait, 30))
                    # Tekrar dene
                    res = requests.post(
                        f"{HF_API}/{model}",
                        json=payload,
                        headers=headers,
                        timeout=90
                    )

                if res.status_code == 200 and res.headers.get('content-type', '').startswith('image'):
                    result_bytes = res.content
                    break
                elif res.status_code == 401:
                    return jsonify({'error': 'HF token hatalı. Write yetkili token oluşturun.'}), 401
                else:
                    try:
                        last_error = res.json().get('error', f'HTTP {res.status_code}')
                    except:
                        last_error = f'HTTP {res.status_code}'
            except requests.Timeout:
                last_error = f"{model} zaman aşımı"
                continue
            except Exception as e:
                last_error = str(e)
                continue

        if result_bytes:
            img_b64 = base64.b64encode(result_bytes).decode()
            return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64})
        else:
            return jsonify({'error': 'Tüm modeller başarısız: ' + last_error}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
