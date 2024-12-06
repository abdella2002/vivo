from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج
model = load_model('H:\cornea\model\VGG16_best_mode.keras')

# أسماء الفئات
class_names = ["cataract", "diabetic_retinopathy","glaucoma", "normal" ]

# دالة لتحضير الصورة
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # حفظ الملف مؤقتًا
    file_path = f"temp_{file.filename}"
    file.save(file_path)
    
    try:
        # تحضير الصورة
        img = preprocess_image(file_path)
        
        # إجراء التنبؤ
        predictions = model.predict(img)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = f"{np.max(predictions) * 100:.2f}%"
        
        # حذف الملف المؤقت
        os.remove(file_path)
        
        # إعادة النتائج
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        # حذف الملف المؤقت إذا حدث خطأ
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)
