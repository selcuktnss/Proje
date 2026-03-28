from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# 📁 Ana klasör yolu
base_dir = os.path.dirname(os.path.abspath(__file__))

# 🤖 Model yükleme
model = tf.keras.models.load_model(os.path.join(base_dir, "model", "fracture_model.h5"))

# 🔍 Tahmin fonksiyonu
def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    # 🔥 threshold biraz düşürdük (daha dengeli)
    if pred > 0.4:
        label = "Kırık Yok"
        confidence = pred
    else:
        label = "Kırık Var"
        confidence = 1 - pred

    return label, round(confidence * 100, 2)

# 🏠 Ana sayfa
@app.route("/", methods=["GET","POST"])
def index():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename.replace(" ", "_")

        static_path = os.path.join(base_dir, "static")
        os.makedirs(static_path, exist_ok=True)

        path = os.path.join(static_path, filename)
        file.save(path)

        result, confidence = predict(path)
        img_path = filename

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path
    )

# ✅ KOD SAYFASI (HATA DÜZELTİLDİ)
@app.route("/code")
def code():
    app_path = os.path.join(base_dir, "app.py")
    train_path = os.path.join(base_dir, "train.py")

    # dosya yoksa hata verme
    app_code = ""
    train_code = ""

    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            app_code = f.read()

    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_code = f.read()

    return render_template("code.html", app_code=app_code, train_code=train_code)

if __name__ == "__main__":
    app.run(debug=True)