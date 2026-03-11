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
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (w//8, h//8, w*6//8, h*6//8)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==1)|(mask==3), 255, 0).astype(np.uint8)
    except:
        mask = np.zeros((h, w), np.uint8)
        mask[h//6:h*5//6, w//6:w*5//6] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask

def apply_pattern(product, pattern, strength=0.9):
    h, w = product.shape[:2]
    ph, pw = pattern.shape[:2]
    reps_x = int(np.ceil(w / pw))
    reps_y = int(np.ceil(h / ph))
    pattern = np.tile(pattern, (reps_y, reps_x, 1))
    pattern = pattern[:h, :w]

    mask = get_foreground_mask(product).astype(np.float32) / 255.0
    mask = np.stack([mask, mask, mask], axis=2)

    product_f = product.astype(np.float32) / 255.0
    pattern_f = pattern.astype(np.float32) / 255.0

    # Koltuğun ışık/gölge haritası — normalize edilmiş
    gray = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Kontrastı artır — ortalamaya göre normalize et
    gray_mean = gray.mean()
    gray_normalized = (gray - gray_mean) * 0.5 + 0.5  # 0.25 - 0.75 arası
    gray_normalized = np.clip(gray_normalized, 0.2, 1.0)
    gray3 = np.stack([gray_normalized, gray_normalized, gray_normalized], axis=2)

    # Desen * ışık — ama deseni karartmadan
    textured = pattern_f * gray3 * 1.8
    textured = np.clip(textured, 0, 1)

    # Blend
    result = product_f * (1 - mask * strength) + textured * (mask * strength)
    return (result * 255).clip(0, 255).astype(np.uint8)

@app.route("/")
def home():
    return "Desen Giydirme API OK"

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data     = request.json
        surf_url = data.get("surf_image") or data.get("product")
        pat_url  = data.get("pat_image") or data.get("pattern")
        strength = float(data.get("strength", 0.9))
        if not surf_url:
            return jsonify({"error": "Ürün görseli eksik"}), 400
        if not pat_url:
            return jsonify({"error": "Desen görseli eksik"}), 400
        product = dataurl_to_cv2(surf_url)
        pattern = dataurl_to_cv2(pat_url)
        h, w = product.shape[:2]
        if max(h, w) > 1024:
            ratio = 1024 / max(h, w)
            product = cv2.resize(product, (int(w*ratio), int(h*ratio)))
        result = apply_pattern(product, pattern, strength)
        return jsonify({"success": True, "image": cv2_to_dataurl(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/dress", methods=["POST"])
def dress():
    return generate()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
