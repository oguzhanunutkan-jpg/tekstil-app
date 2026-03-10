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
        api_key  = data.get('api_key', '').strip()
        surf_url = data.get('surf_image')
        pat_url  = data.get('pat_image')
        prompt   = data.get('prompt', 'Apply an elegant fabric pattern to the upholstery')

        if not api_key:
            return jsonify({'error': 'Gemini API key gerekli'}), 400
        if not surf_url:
            return jsonify({'error': 'Ürün görseli eksik'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 512)
        surf_b64   = base64.b64encode(surf_bytes).decode()

        parts = [
            {
                "text": f"{prompt}. Keep the exact shape, form and structure of the furniture. Only change the fabric/upholstery texture. Make it look realistic and professional."
            },
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": surf_b64
                }
            }
        ]

        # Desen görseli de varsa ekle
        if pat_url:
            pat_bytes = resize_image(dataurl_to_bytes(pat_url), 256)
            pat_b64   = base64.b64encode(pat_bytes).decode()
            parts.append({
                "text": "Use this pattern for the fabric:"
            })
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": pat_b64
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }

        res = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json=payload,
            timeout=60
        )

        if res.status_code == 400:
            return jsonify({'error': 'API key hatalı veya istek geçersiz.'}), 400
        if res.status_code == 403:
            return jsonify({'error': 'API key yetkisiz. Google AI Studio\'dan alın.'}), 403
        if res.status_code != 200:
            try:
                msg = res.json().get('error', {}).get('message', res.text[:200])
            except:
                msg = res.text[:200]
            return jsonify({'error': f'HTTP {res.status_code}: {msg}'}), 500

        # Yanıttan görsel çıkar
        candidates = res.json().get('candidates', [])
        for candidate in candidates:
            for part in candidate.get('content', {}).get('parts', []):
                if 'inlineData' in part:
                    img_data = part['inlineData']['data']
                    mime     = part['inlineData'].get('mimeType', 'image/png')
                    return jsonify({'success': True, 'image': f'data:{mime};base64,{img_data}'})

        return jsonify({'error': 'Görsel üretilemedi. Prompt değiştirip tekrar deneyin.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
