import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from django.conf import settings
from .gradcam_utils import make_gradcam_heatmap

# Load model ONCE (important for Django performance)
MODEL_PATH = os.path.join(settings.BASE_DIR, "skin_cancer_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Automatically find last conv layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if len(layer.output.shape) == 4:
        last_conv_layer_name = layer.name
        break


def predict_skin_cancer(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediction
    pred = model.predict(img_array)[0][0]
    label = "Melanoma" if pred > 0.5 else "Benign"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    # 🔥 GRAD-CAM (FIXED CALL)
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    # Save heatmap image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(settings.MEDIA_ROOT, "heatmap.jpg")
    overlay_path = os.path.join(settings.MEDIA_ROOT, "overlay.jpg")

    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    return label, confidence, "/media/heatmap.jpg", "/media/overlay.jpg"