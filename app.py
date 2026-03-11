import os, base64, io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# -------------------------
# Utils
# -------------------------

def dataurl_to_cv2(dataurl):
    _, encoded = dataurl.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_dataurl(img):
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode()
    return "data:image/png;base64," + b64

def resize_if_large(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        ratio = max_size / max(h, w)
        img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
    return img

# -------------------------
# Pattern tile (repeat)
# -------------------------

def tile_pattern(pattern, target_w, target_h):

    ph, pw = pattern.shape[:2]

    reps_x = int(np.ceil(target_w / pw))
    reps_y = int(np.ceil(target_h / ph))

    tiled = np.tile(pattern, (reps_y, reps_x, 1))
    return tiled[:target_h, :target_w]

# -------------------------
# Foreground mask
# -------------------------

def get_foreground_mask(img):

    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rect = (w//8, h//8, w*6//8, h*6//8)

    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==1) | (mask==3), 255, 0).astype(np.uint8)
    except:
        mask = np.zeros((h,w),np.uint8)
        mask[h//6:h*5//6, w//6:w*5//6] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask,(21,21),0)

    return mask

# -------------------------
# Pattern apply
# -------------------------

def apply_pattern(product, pattern, strength=0.8):

    h, w = product.shape[:2]

    pattern_tiled = tile_pattern(pattern, w, h)

    fg_mask = get_foreground_mask(product)
    mask_norm = fg_mask.astype(np.float32) / 255.0

    product_f = product.astype(np.float32) / 255
    pattern_f = pattern_tiled.astype(np.float32) / 255

    # Multiply blend (kumaş efekti)
    blended = product_f * pattern_f * 1.8
    blended = np.clip(blended, 0, 1)

    result = product_f.copy()

    for c in range(3):
        result[:,:,c] = (
            product_f[:,:,c]*(1-mask_norm*strength)
            + blended[:,:,c]*(mask_norm*strength)
        )

    return (result*255).astype(np.uint8)

# -------------------------
# API
# -------------------------

@app.route("/")
def home():
    return "Desen Giydirme API OK"

@app.route("/api/generate", methods=["POST"])
def generate():

    try:

        data = request.json

        surf = data.get("surf_image") or data.get("product")
        pat  = data.get("pat_image") or data.get("pattern")
        strength = float(data.get("strength",0.8))

        if not surf:
            return jsonify({"error":"Ürün görseli eksik"}),400

        if not pat:
            return jsonify({"error":"Desen görseli eksik"}),400

        product = resize_if_large(dataurl_to_cv2(surf))
        pattern = resize_if_large(dataurl_to_cv2(pat))

        result = apply_pattern(product, pattern, strength)

        return jsonify({
            "success":True,
            "image":cv2_to_dataurl(result)
        })

    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/dress", methods=["POST"])
def dress():
    return generate()

if __name__ == "__main__":

    port = int(os.environ.get("PORT",5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
