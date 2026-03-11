import os, base64, io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

def dataurl_to_cv2(dataurl):
    _, encoded = dataurl.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_dataurl(img):
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode()
    return "data:image/png;base64," + b64

def auto_mask(product):
    """Ürünün ön planını otomatik maske olarak çıkar"""
    gray = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # GrabCut ile ön plan tespiti
    h, w = product.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (w//10, h//10, w*8//10, h*8//10)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    
    try:
        cv2.grabCut(product, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
    except:
        # Fallback: merkez bölge
        mask2 = np.zeros((h, w), np.uint8)
        mask2[h//8:h*7//8, w//8:w*7//8] = 255
    
    # Kenarları yumuşat
    mask2 = cv2.GaussianBlur(mask2, (21,21), 0)
    return mask2

def apply_pattern(product, pattern, strength=0.85):
    h, w = product.shape[:2]
    pattern_resized = cv2.resize(pattern, (w, h))
    
    # Otomatik mask
    mask = auto_mask(product)
    mask_norm = mask / 255.0
    
    # Blend modları — multiply + overlay karışımı
    product_f = product.astype(np.float32) / 255.0
    pattern_f = pattern_resized.astype(np.float32) / 255.0
    
    # Multiply blend
    multiply = product_f * pattern_f
    
    # Overlay blend
    overlay = np.where(
        product_f < 0.5,
        2 * product_f * pattern_f,
        1 - 2 * (1 - product_f) * (1 - pattern_f)
    )
    
    # İkisini karıştır
    blended = multiply * 0.5 + overlay * 0.5
    
    # Orijinalle karıştır (strength kontrolü)
    result_f = product_f * (1 - mask_norm[:,:,np.newaxis] * strength) + \
               blended * (mask_norm[:,:,np.newaxis] * strength)
    
    result = (result_f * 255).clip(0, 255).astype(np.uint8)
    return result

@app.route("/")
def home():
    return "Desen Giydirme API calisiyor OK"

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data     = request.json
        surf_url = data.get("surf_image") or data.get("product")
        pat_url  = data.get("pat_image") or data.get("pattern")
        strength = float(data.get("strength", 0.75))

        if not surf_url:
            return jsonify({"error": "Ürün görseli eksik"}), 400
        if not pat_url:
            return jsonify({"error": "Desen görseli eksik"}), 400

        product = dataurl_to_cv2(surf_url)
        pattern = dataurl_to_cv2(pat_url)

        result = apply_pattern(product, pattern, strength)
        return jsonify({"success": True, "image": cv2_to_dataurl(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Eski endpoint de çalışsın
@app.route("/api/dress", methods=["POST"])
def dress():
    return generate()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
