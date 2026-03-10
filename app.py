import os, base64, io, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import replicate

app = Flask(__name__)
CORS(app)  # Tüm originlere izin ver

def dataurl_to_bytes(dataurl):
    """Base64 data URL'yi bytes'a çevir"""
    header, data = dataurl.split(',', 1)
    return base64.b64decode(data)

def resize_image(img_bytes, max_size=768):
    """Görseli yeniden boyutlandır"""
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
    return "DeseniGiydır API çalışıyor ✓"

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        surf_dataurl = data.get('surf_image')
        pat_dataurl = data.get('pat_image')
        prompt = data.get('prompt', 'office chair with elegant fabric pattern, realistic product photo')
        strength = float(data.get('strength', 0.7))

        if not api_key:
            return jsonify({'error': 'Replicate API key gerekli'}), 400
        if not surf_dataurl or not pat_dataurl:
            return jsonify({'error': 'Görseller eksik'}), 400

        # Görselleri boyutlandır
        surf_bytes = resize_image(dataurl_to_bytes(surf_dataurl), 768)
        pat_bytes  = resize_image(dataurl_to_bytes(pat_dataurl), 512)

        # Replicate client
        client = replicate.Client(api_token=api_key)

        # SDXL img2img modeli
        output = client.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "image": io.BytesIO(surf_bytes),
                "prompt": prompt + ", realistic fabric texture, professional product photography, high quality, detailed",
                "negative_prompt": "blurry, low quality, distorted, deformed, cartoon, watermark",
                "prompt_strength": strength,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "scheduler": "K_EULER",
            }
        )

        # Sonuç URL'si
        if output and len(output) > 0:
            result_url = output[0] if isinstance(output[0], str) else str(output[0])
            # URL'den base64'e çevir (CORS sorununu önlemek için)
            img_resp = requests.get(result_url, timeout=30)
            img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
            return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64})
        else:
            return jsonify({'error': 'Sonuç alınamadı'}), 500

    except replicate.exceptions.ReplicateError as e:
        msg = str(e)
        if '401' in msg or 'Unauthenticated' in msg:
            return jsonify({'error': 'API key hatalı veya geçersiz'}), 401
        if '402' in msg or 'payment' in msg.lower():
            return jsonify({'error': 'Replicate krediniz bitti'}), 402
        return jsonify({'error': 'Replicate hatası: ' + msg}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_key():
    """API key'i test et"""
    try:
        data = request.json
        api_key = data.get('api_key', '').strip()
        client = replicate.Client(api_token=api_key)
        # Basit bir model listesi çek
        account = client.models.get("stability-ai/sdxl")
        return jsonify({'success': True, 'message': 'API key geçerli ✓'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
