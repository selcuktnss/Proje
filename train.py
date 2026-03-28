import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 📁 klasör yolları
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "model")
static_dir = os.path.join(base_dir, "static")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

data_dir = os.path.join(base_dir, "dataset")

# 📊 veri artırma
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.3,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# 🔥 MobileNetV2 (CNN)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# son katmanları aç (fine-tuning)
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ⚖️ class weight
class_weight = {0:1.0, 1:1.5}

# 🛑 early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 🚀 eğitim
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=class_weight,
    callbacks=[early_stop]
)

# 💾 model kaydet
model.save(os.path.join(model_dir, "fracture_model.h5"))

print("MODEL KAYDEDİLDİ ✅")

# 📊 ACCURACY
plt.figure()
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(static_dir, "accuracy.png"))
plt.close()

# 📊 LOSS
plt.figure()
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(static_dir, "loss.png"))
plt.close()

# 📊 CONFUSION MATRIX
val_data.reset()
preds = model.predict(val_data)
preds = (preds > 0.4).astype(int).reshape(-1)

cm = confusion_matrix(val_data.classes, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(static_dir, "confusion_matrix.png"))
plt.close()

print("GRAFİKLER HAZIR ✅")