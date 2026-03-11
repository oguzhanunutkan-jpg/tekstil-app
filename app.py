import os, base64, io, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from gradio_client import Client, handle_file

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
        surf_url = data.get('surf_image')
        prompt   = data.get('prompt', 'office chair with elegant fabric pattern, realistic product photo')
        strength = float(data.get('strength', 0.7))

        if not surf_url:
            return jsonify({'error': 'Görsel eksik'}), 400

        surf_bytes = resize_image(dataurl_to_bytes(surf_url), 512)
        full_prompt = prompt + ", realistic fabric texture, professional product photography, high quality"

        # Geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(surf_bytes)
            tmp_path = tmp.name

        # Gradio Client — endpoint otomatik bulur
        client = Client("nightfury/StableDiffusion.Img2Img-Gradio")
        result = client.predict(
            handle_file(tmp_path),  # image
            full_prompt,            # prompt
            "blurry, low quality, distorted, cartoon, watermark",  # negative
            20,                     # steps
            strength,               # strength
            7.5,                    # guidance scale
            api_name="/predict"
        )

        # Sonuç dosya yolu
        if isinstance(result, str) and os.path.exists(result):
            with open(result, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64})
        elif isinstance(result, dict) and 'path' in result:
            with open(result['path'], 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return jsonify({'success': True, 'image': 'data:image/png;base64,' + img_b64})

        return jsonify({'error': 'Beklenmedik sonuç: ' + str(result)[:200]}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
