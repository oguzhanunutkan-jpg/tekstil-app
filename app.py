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

def get_foreground_mask(img):
    h, w = img.shape[:2]
    
    # GrabCut — merkez dikdörtgen (koltuk genellikle ortada)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    
    # Kenarlardan içeri al — arka planı dışarıda bırak
    margin_x = w // 6
    margin_y = h // 6
    rect = (margin_x, margin_y, w - margin_x*2, h - margin_y*2)
    
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    except:
        fg_mask = np.zeros((h, w), np.uint8)
        fg_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255
    
    # Morfolojik işlem — delikler kapat, kenarları düzelt
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    
    # Kenar yumuşat
    fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)
    
    return fg_mask

def apply_pattern(product, pattern, strength=0.80):
    h, w = product.shape[:2]
    pattern_resized = cv2.resize(pattern, (w, h))
    
    # Sadece ön plan maskesi
    fg_mask = get_foreground_mask(product)
    mask_norm = fg_mask.astype(np.float32) / 255.0
    
    product_f = product.astype(np.float32) / 255.0
    pattern_f = pattern_resized.astype(np.float32) / 255.0
    
    # Multiply blend — kumaş dokusunu korur
    multiply = product_f * pattern_f * 2.0
    multiply = np.clip(multiply, 0, 1)
    
    # Overlay blend
    overlay = np.where(
        product_f < 0.5,
        2 * product_f * pattern_f,
        1 - 2 * (1 - product_f) * (1 - pattern_f)
    )
    
    blended = multiply * 0.4 + overlay * 0.6
    
    # Sadece ön planda uygula — arka plan orijinal kalsın
    result_f = product_f.copy()
    for c in range(3):
        result_f[:,:,c] = (
            product_f[:,:,c] * (1 - mask_norm * strength) +
            blended[:,:,c] * (mask_norm * strength)
        )
    
    return (result_f * 255).clip(0, 255).astype(np.uint8)

@app.route("/")
def home():
    return "Desen Giydirme API calisiyor OK"

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data     = request.json
        surf_url = data.get("surf_image") or data.get("product")
        pat_url  = data.get("pat_image") or data.get("pattern")
        strength = float(data.get("strength", 0.80))

        if not surf_url:
            return jsonify({"error": "Ürün görseli eksik"}), 400
        if not pat_url:
            return jsonify({"error": "Desen görseli eksik"}), 400

        product = dataurl_to_cv2(surf_url)
        pattern = dataurl_to_cv2(pat_url)
        result  = apply_pattern(product, pattern, strength)

        return jsonify({"success": True, "image": cv2_to_dataurl(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/dress", methods=["POST"])
def dress():
    return generate()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
