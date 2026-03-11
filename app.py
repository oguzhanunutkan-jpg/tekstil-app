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

def tile_pattern(pattern, w, h):
    ph, pw = pattern.shape[:2]
    reps_x = int(np.ceil(w / pw))
    reps_y = int(np.ceil(h / ph))
    tiled = np.tile(pattern, (reps_y, reps_x, 1))
    return tiled[:h, :w]

def get_sofa_mask(img):
    """Koltuk kumaş bölgesini tespit et"""
    h, w = img.shape[:2]
    
    # 1. GrabCut ile ön plan al
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    # Dikdörtgeni daralt — kenar bölgeleri arka plan
    rect = (w//10, h//10, w*8//10, h*8//10)
    try:
        cv2.grabCut(img, mask_gc, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask_gc==1)|(mask_gc==3), 255, 0).astype(np.uint8)
    except:
        fg = np.zeros((h,w), np.uint8)
        fg[h//8:h*7//8, w//8:w*7//8] = 255

    # 2. Renk bazlı filtre — koltuk rengi arka plandan farklı
    # Koltuk merkezinin rengini al
    cy, cx = h//2, w//2
    center_color = img[cy-20:cy+20, cx-20:cx+20].mean(axis=(0,1))
    
    # HSV'ye çevir
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center_hsv = cv2.cvtColor(
        np.uint8([[center_color]]), cv2.COLOR_BGR2HSV
    )[0][0]
    
    # Merkez rengine yakın pikselleri bul
    h_range = 25
    s_range = 60
    v_range = 80
    lower = np.array([
        max(0,  int(center_hsv[0])-h_range),
        max(0,  int(center_hsv[1])-s_range),
        max(0,  int(center_hsv[2])-v_range)
    ])
    upper = np.array([
        min(180, int(center_hsv[0])+h_range),
        min(255, int(center_hsv[1])+s_range),
        min(255, int(center_hsv[2])+v_range)
    ])
    color_mask = cv2.inRange(hsv, lower, upper)
    
    # GrabCut + renk maskesini birleştir
    combined = cv2.bitwise_and(fg, color_mask)
    
    # Morfoloji — boşlukları doldur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    
    # En büyük bileşeni al (koltuk gövdesi)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        combined = np.where(labels == largest, 255, 0).astype(np.uint8)
    
    # Yumuşak kenar
    combined = cv2.GaussianBlur(combined, (25, 25), 0)
    return combined

def apply_pattern(product, pattern, strength=0.85):
    h, w = product.shape[:2]
    pattern_tiled = tile_pattern(pattern, w, h)
    
    mask = get_sofa_mask(product)
    mask_norm = mask.astype(np.float32) / 255.0
    
    p = product.astype(np.float32) / 255.0
    t = pattern_tiled.astype(np.float32) / 255.0
    
    # Soft light blend — kumaş dokusu gibi görünür
    soft_light = np.where(
        t <= 0.5,
        p - (1 - 2*t) * p * (1-p),
        p + (2*t - 1) * (np.sqrt(p) - p)
    )
    soft_light = np.clip(soft_light, 0, 1)
    
    # Multiply ile karıştır
    multiply = p * t * 1.6
    multiply = np.clip(multiply, 0, 1)
    
    blended = soft_light * 0.6 + multiply * 0.4
    
    # Sadece maske bölgesine uygula
    result = p.copy()
    for c in range(3):
        result[:,:,c] = (
            p[:,:,c] * (1 - mask_norm * strength) +
            blended[:,:,c] * (mask_norm * strength)
        )
    
    return (result * 255).clip(0,255).astype(np.uint8)

@app.route("/")
def home():
    return "Desen Giydirme API OK"

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data     = request.json
        surf_url = data.get("surf_image") or data.get("product")
        pat_url  = data.get("pat_image") or data.get("pattern")
        strength = float(data.get("strength", 0.85))

        if not surf_url:
            return jsonify({"error": "Ürün görseli eksik"}), 400
        if not pat_url:
            return jsonify({"error": "Desen görseli eksik"}), 400

        product = dataurl_to_cv2(surf_url)
        pattern = dataurl_to_cv2(pat_url)
        
        # Büyük görselleri küçült
        max_size = 1024
        h, w = product.shape[:2]
        if max(h,w) > max_size:
            ratio = max_size / max(h,w)
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
