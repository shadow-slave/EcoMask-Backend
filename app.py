import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv          # <--- NEW IMPORT
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

app = Flask(__name__)

# --- 1. SECURE DATABASE CONNECTION ---
print("Loading environment variables...")
load_dotenv()  # <--- Loads the .env file

print("Connecting to MongoDB...")
# Securely fetch the URI from the environment
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("FATAL ERROR: MONGO_URI is missing from the environment!")

client = MongoClient(MONGO_URI)
db = client.ecomask_db
scans_collection = db.scans
print("MongoDB Connected Securely!")

# --- 2. LOAD THE AI BRAIN ---
print("Booting up EcoMask AI...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
cfg.MODEL.WEIGHTS = "./model_final.pth" 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
print("EcoMask AI is online and ready!")


# ==========================================
# ROUTE 1: ANALYZE ONLY (DOES NOT SAVE)
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    file_bytes = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    class_ids = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    
    class_names = ["Plastic", "Metal", "Paper", "Glass", "Bio/Unclassified"]
    detected_items = [{"class": class_names[cid], "confidence": round(score, 2)} for cid, score in zip(class_ids, scores)]
    
    # Notice: No MongoDB code here! It just returns the answer.
    return jsonify({
        "message": "Analysis complete",
        "total_items_found": len(detected_items),
        "items": detected_items
    })


# ==========================================
# ROUTE 2: SUBMIT REPORT (SAVES TO MONGODB)
# ==========================================
@app.route('/submit-report', methods=['POST'])
def submit_report():
    # The mobile app will send a JSON payload with the data it wants to report
    data = request.json
    
    if not data or 'items' not in data or 'lat' not in data or 'lng' not in data:
        return jsonify({"error": "Missing required report data (items, lat, lng)"}), 400

    scan_record = {
        "timestamp": datetime.utcnow(),
        "location": {
            "type": "Point",
            "coordinates": [data['lng'], data['lat']] # MongoDB requires Longitude first
        },
        "total_items_reported": len(data['items']),
        "items": data['items']
    }
    
    scans_collection.insert_one(scan_record)
    
    return jsonify({
        "message": "Report successfully submitted to the municipal dashboard!",
        "record_id": str(scan_record["_id"])
    })


# ==========================================
# ROUTE 3: FETCH HEATMAP DATA
# ==========================================
@app.route('/heatmap-data', methods=['GET'])
def get_heatmap_data():
    scans = list(scans_collection.find({}, {"_id": 0}))
    return jsonify(scans)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)