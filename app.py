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

        if not hf_token:
            return jsonify({'error': 'Hugging Face token gerekli'}), 400
        if not surf_url:
            return jsonify({'error': 'Görsel eksik'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 512)
        full_prompt = prompt + ", realistic fabric texture, professional product photography, high quality"

        headers = {"Authorization": f"Bearer {hf_token}"}

        # Yeni HF router endpoint — multipart olarak gönder
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/timbrooks/instruct-pix2pix/v1/image-to-image",
            headers=headers,
            files={"image": ("image.jpg", surf_bytes, "image/jpeg")},
            data={
                "prompt": full_prompt,
                "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, watermark",
                "num_inference_steps": "20",
                "guidance_scale": "7.5",
                "image_guidance_scale": "1.5",
            },
            timeout=120
        )

        if response.status_code == 401:
            return jsonify({'error': 'Token hatalı. Write yetkili HF token girin.'}), 401
        if response.status_code == 503:
            return jsonify({'error': 'Model yükleniyor, 30 saniye sonra tekrar deneyin.'}), 503
        if response.status_code == 404:
            return jsonify({'error': 'Model bulunamadı.'}), 404

        content_type = response.headers.get('content-type', '')
        if response.status_code == 200 and 'image' in content_type:
            img_b64 = base64.b64encode(response.content).decode()
            return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + img_b64})

        # Hata detayı göster
        try:
            msg = response.json().get('error', response.text[:300])
        except:
            msg = response.text[:300]
        return jsonify({'error': f'HTTP {response.status_code}: {msg}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
