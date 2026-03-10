import os, base64, io, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

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
        strength = float(data.get('strength', 0.7))

        if not surf_url:
            return jsonify({'error': 'Görsel eksik'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 512)

        # Görseli geçici olarak kaydet ve URL oluştur
        img_b64 = base64.b64encode(surf_bytes).decode()
        
        full_prompt = prompt + ", realistic fabric texture, professional product photography, high quality"

        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # fffiloni/stable-diffusion-img2img Gradio API
        payload = {
            "data": [
                f"data:image/jpeg;base64,{img_b64}",  # image
                full_prompt,                            # prompt
                "blurry, low quality, distorted, deformed, cartoon, watermark",  # negative
                8,   # steps
                strength,  # strength
                7.5  # guidance scale
            ]
        }

        res = requests.post(
            "https://fffiloni-stable-diffusion-img2img.hf.space/gradio_api/predict",
            json=payload,
            headers=headers,
            timeout=120
        )

        if res.status_code != 200:
            try:
                msg = res.json()
            except:
                msg = res.text[:300]
            return jsonify({'error': f'HTTP {res.status_code}: {msg}'}), 500

        result = res.json()
        output_data = result.get('data', [])
        
        if output_data and len(output_data) > 0:
            img_data = output_data[0]
            if isinstance(img_data, str) and img_data.startswith('data:'):
                return jsonify({'success': True, 'image': img_data})
            elif isinstance(img_data, dict) and 'url' in img_data:
                img_url = img_data['url']
                if img_url.startswith('/'):
                    img_url = 'https://fffiloni-stable-diffusion-img2img.hf.space' + img_url
                img_res = requests.get(img_url, timeout=30)
                img_b64_out = base64.b64encode(img_res.content).decode()
                return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64_out})

        return jsonify({'error': 'Sonuç alınamadı: ' + str(result)[:200]}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
