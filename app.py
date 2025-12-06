from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global variables to store trained models
svm_model = None
rf_model = None
last_metrics = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """Enhance image using adaptive histogram equalization"""
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        img = decode_image(image_data)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = img.copy()
        for i in range(3):
            enhanced[:,:,i] = clahe.apply(img[:,:,i])
        
        # Encode back to base64
        enhanced_b64 = encode_image(enhanced)
        
        return jsonify({
            'success': True,
            'enhanced_image': enhanced_b64
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect', methods=['POST'])
def detect_fire():
    """Detect fire regions in the image"""
    try:
        data = request.json
        image_data = data['image']
        sensitivity = data.get('sensitivity', 0.5)
        min_area = data.get('min_area', 300)
        
        # Decode image
        img = decode_image(image_data)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]
        
        # Fire detection based on HSV thresholds
        # Fire typically has hue in red range (0-30 degrees in 0-180 scale)
        mask1 = (H >= 0) & (H <= 0.15) & (S > 0.4 * sensitivity) & (V > 0.5 * sensitivity)
        
        # Additional RGB-based detection for red dominance
        R = img[:,:,2].astype(np.float32)
        G = img[:,:,1].astype(np.float32)
        B = img[:,:,0].astype(np.float32)
        red_dominant = (R > 1.1 * G) & (R > 1.1 * B)
        
        # Combine masks
        mask = (mask1 | red_dominant).astype(np.uint8)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask)
        regions = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 1
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                regions.append((x, y, w, h))
        
        # Calculate fire percentage
        total_fire_pixels = np.sum(cleaned_mask)
        total_pixels = cleaned_mask.size
        fire_percent = (total_fire_pixels / total_pixels) * 100
        
        # Create overlay
        overlay = img.copy()
        overlay[:,:,1] = (overlay[:,:,1] * 0.3).astype(np.uint8)
        overlay[:,:,0] = (overlay[:,:,0] * 0.3).astype(np.uint8)
        overlay[:,:,2] = np.clip(overlay[:,:,2] + 120 * cleaned_mask, 0, 255).astype(np.uint8)
        
        # Blend
        alpha = 0.6
        blended = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        
        # Draw bounding boxes
        for (x, y, w, h) in regions:
            cv2.rectangle(blended, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # Encode result
        result_b64 = encode_image(blended)
        
        return jsonify({
            'success': True,
            'result_image': result_b64,
            'fire_percent': float(fire_percent),
            'num_regions': len(regions)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_ml', methods=['POST'])
def predict_ml():
    """Predict fire/no fire using trained ML model"""
    try:
        if rf_model is None:
            return jsonify({'success': False, 'error': 'Model not trained yet. Click "Train+Evaluate ML" first.'})
        
        data = request.json
        image_data = data['image']
        
        # Decode image
        img = decode_image(image_data)
        
        # Extract features (same as training!)
        features = extract_features(img)
        features = np.array(features).reshape(1, -1)
        
        # Predict
        pred = rf_model.predict(features)[0]
        prob = rf_model.predict_proba(features)[0]
        
        label = "Fire" if pred == 1 else "No Fire"
        confidence = float(prob[pred])
        
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': round(confidence, 3)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/train', methods=['POST'])
def train_models():
    """Train SVM and Random Forest models"""
    try:
        # Build dataset
        X, Y = build_dataset()
        
        if len(X) == 0:
            return jsonify({
                'success': False,
                'error': 'Dataset missing: Create plantvillage/fire and plantvillage/no fire folders with images'
            })
        
        # Split dataset
        if len(X) < 5:
            X_train, X_test, Y_train, Y_test = X, X, Y, Y
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Train SVM
        global svm_model, rf_model
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(X_train, Y_train)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=60, random_state=42)
        rf_model.fit(X_train, Y_train)
        
        # Predictions
        Y_pred_svm = svm_model.predict(X_test)
        Y_pred_rf = rf_model.predict(X_test)
        
        # Get probabilities for ROC
        svm_probs = svm_model.predict_proba(X_test)[:, 1]
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc_svm = accuracy_score(Y_test, Y_pred_svm) * 100
        acc_rf = accuracy_score(Y_test, Y_pred_rf) * 100
        
        prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(Y_test, Y_pred_svm, average='binary')
        prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(Y_test, Y_pred_rf, average='binary')
        
        # ROC curves
        fpr_svm, tpr_svm, _ = roc_curve(Y_test, svm_probs)
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, rf_probs)
        auc_svm = auc(fpr_svm, tpr_svm)
        auc_rf = auc(fpr_rf, tpr_rf)
        
        
        # Plot ROC curves
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_svm, tpr_svm, linewidth=2, label=f'SVM (AUC = {auc_svm:.2f})')
        plt.plot(fpr_rf, tpr_rf, linewidth=2, label=f'RF (AUC = {auc_rf:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - SVM vs Random Forest')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('static/roc_curve.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Confusion matrices
        cm_svm = confusion_matrix(Y_test, Y_pred_svm)
        cm_rf = confusion_matrix(Y_test, Y_pred_rf)
        
        # Plot confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].imshow(cm_svm, cmap='Blues')
        axes[0].set_title('SVM Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, cm_svm[i, j], ha='center', va='center')
        
        im2 = axes[1].imshow(cm_rf, cmap='Blues')
        axes[1].set_title('RF Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, cm_rf[i, j], ha='center', va='center')
        
        plt.savefig('static/confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        message = f"""✅ ML Evaluation Complete
SVM → Acc: {acc_svm:.2f}% | Prec: {prec_svm:.2f} | Rec: {rec_svm:.2f} | F1: {f1_svm:.2f} | AUC: {auc_svm:.2f}
RF  → Acc: {acc_rf:.2f}% | Prec: {prec_rf:.2f} | Rec: {rec_rf:.2f} | F1: {f1_rf:.2f} | AUC: {auc_rf:.2f}"""
        
        return jsonify({
            'success': True,
            'message': message,
            'metrics': {
                'svm': {'acc': acc_svm, 'prec': prec_svm, 'rec': rec_svm, 'f1': f1_svm, 'auc': auc_svm},
                'rf': {'acc': acc_rf, 'prec': prec_rf, 'rec': rec_rf, 'f1': f1_rf, 'auc': auc_rf}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/metrics')
def get_metrics():
    """Return precomputed metrics for display"""
    global last_metrics
    if last_metrics is None:
        try:
            X, Y = build_dataset()
            if len(X) == 0:
                return jsonify({'error': 'No dataset'})
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            temp_svm = SVC(kernel='rbf', probability=True, random_state=42)
            temp_svm.fit(X_train, Y_train)
            Y_pred_svm = temp_svm.predict(X_test)
            acc_svm = accuracy_score(Y_test, Y_pred_svm) * 100
            prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(Y_test, Y_pred_svm, average='binary')
            fpr_svm, tpr_svm, _ = roc_curve(Y_test, temp_svm.predict_proba(X_test)[:,1])
            auc_svm = auc(fpr_svm, tpr_svm)
            
            temp_rf = RandomForestClassifier(n_estimators=60, random_state=42)
            temp_rf.fit(X_train, Y_train)
            Y_pred_rf = temp_rf.predict(X_test)
            acc_rf = accuracy_score(Y_test, Y_pred_rf) * 100
            prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(Y_test, Y_pred_rf, average='binary')
            fpr_rf, tpr_rf, _ = roc_curve(Y_test, temp_rf.predict_proba(X_test)[:,1])
            auc_rf = auc(fpr_rf, tpr_rf)
            
            last_metrics = {
                'svm': {'acc': acc_svm, 'prec': prec_svm, 'rec': rec_svm, 'f1': f1_svm, 'auc': auc_svm},
                'rf': {'acc': acc_rf, 'prec': prec_rf, 'rec': rec_rf, 'f1': f1_rf, 'auc': auc_rf}
            }
        except Exception as e:
            return jsonify({'error': f'Metrics computation failed: {str(e)}'})
    
    return jsonify(last_metrics)

def build_dataset():
    """Build feature dataset from fire and no fire images"""
    root = 'plantvillage'
    fire_folder = os.path.join(root, 'fire')
    nofire_folder = os.path.join(root, 'no fire')
    
    X = []
    Y = []
    
    # Load fire images
    if os.path.exists(fire_folder):
        for filename in os.listdir(fire_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(fire_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    features = extract_features(img)
                    X.append(features)
                    Y.append(1)
    
    # Load no fire images
    if os.path.exists(nofire_folder):
        for filename in os.listdir(nofire_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(nofire_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    features = extract_features(img)
                    X.append(features)
                    Y.append(0)
    
    print(f"✅ Loaded {len(Y)} samples | Fire: {Y.count(1)}, No fire: {Y.count(0)}")
    return np.array(X), np.array(Y)

def extract_features(img):
    """Extract color features from image"""
    # Resize to standard size
    img = cv2.resize(img, (128, 128))
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    
    # HSV statistics
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])
    std_v = np.std(hsv[:,:,2])
    
    # RGB ratios
    R = img[:,:,2].astype(np.float32)
    G = img[:,:,1].astype(np.float32)
    B = img[:,:,0].astype(np.float32)
    
    rg_ratio = np.mean(R / (G + 1))
    rb_ratio = np.mean(R / (B + 1))
    
    return [mean_h, mean_s, mean_v, std_v, rg_ratio, rb_ratio]

def decode_image(base64_string):
    """Decode base64 string to OpenCV image"""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def encode_image(img):
    """Encode OpenCV image to base64 string"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_base64}'

if __name__ == '__main__':
    # Create static folder for plots
    os.makedirs('static', exist_ok=True)
    
    # Pre-train models on startup
    print("🔄 Pre-training ML models on startup...")
    try:
        X, Y = build_dataset()
        if len(X) == 0:
            print("⚠️ Warning: No training data found. Prediction will fail.")
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            # Assign to global variables (no 'global' needed at module level)
            svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            svm_model.fit(X_train, Y_train)
            
            rf_model = RandomForestClassifier(n_estimators=60, random_state=42)
            rf_model.fit(X_train, Y_train)
            
            print(f"✅ Models trained on {len(Y)} samples.")
    except Exception as e:
        print(f"❌ Pre-training failed: {e}")
    
    # Run the app
    if __name__ == '__main__':
        # ... [your pre-training code] ...
        
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)