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

def get_fabric_mask(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Arka plan: çok açık piksel (beyaz/gri arka plan + gölge)
    _, bg_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Çok koyu piksel: ayaklar (siyah/koyu ahşap)
    _, dark_mask = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.bitwise_and(fg_mask, dark_mask)

    # GrabCut
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (w//8, h//10, w*6//8, h*7//10)  # alt kısımı dışarıda bırak
    try:
        cv2.grabCut(img, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        gc_fg = np.where((gc_mask==1)|(gc_mask==3), 255, 0).astype(np.uint8)
        fg_mask = cv2.bitwise_and(fg_mask, gc_fg)
    except:
        pass

    # Alt %20'yi kaldır — ayaklar + gölge genellikle altta
    cutoff = int(h * 0.78)
    fg_mask[cutoff:, :] = 0

    # Morfoloji
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    # En büyük bileşen
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)
    return fg_mask

def apply_pattern(product, pattern, strength=0.9):
    h, w = product.shape[:2]
    ph, pw = pattern.shape[:2]
    reps_x = int(np.ceil(w / pw))
    reps_y = int(np.ceil(h / ph))
    pattern = np.tile(pattern, (reps_y, reps_x, 1))
    pattern = pattern[:h, :w]

    mask = get_fabric_mask(product).astype(np.float32) / 255.0
    mask3 = np.stack([mask, mask, mask], axis=2)

    product_f = product.astype(np.float32) / 255.0
    pattern_f = pattern.astype(np.float32) / 255.0

    gray = cv2.cvtColor(product, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_mean = gray.mean()
    gray_norm = np.clip((gray - gray_mean) * 0.5 + 0.5, 0.2, 1.0)
    gray3 = np.stack([gray_norm, gray_norm, gray_norm], axis=2)

    textured = np.clip(pattern_f * gray3 * 1.8, 0, 1)
    result = product_f * (1 - mask3 * strength) + textured * (mask3 * strength)
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
