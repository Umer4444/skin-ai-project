import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "skin_cancer_model.keras"
IMAGE_PATH = "data/test/melanoma"   # folder path
IMG_SIZE = (224, 224)

# ----------------------------
# Load model
# ----------------------------
model = load_model(MODEL_PATH)

# Find last conv layer automatically
last_conv_layer = None
for layer in reversed(model.layers):
    if len(layer.output.shape) == 4:
        last_conv_layer = layer.name
        break

print("Using last conv layer:", last_conv_layer)

# ----------------------------
# Load one image
# ----------------------------
import os
img_name = os.listdir(IMAGE_PATH)[0]
img_path = os.path.join(IMAGE_PATH, img_name)

img = load_img(img_path, target_size=IMG_SIZE)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ----------------------------
# Grad-CAM Model
# ----------------------------
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ----------------------------
# Overlay heatmap
# ----------------------------
img_original = cv2.imread(img_path)
img_original = cv2.resize(img_original, IMG_SIZE)

heatmap = cv2.resize(heatmap, IMG_SIZE)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

# ----------------------------
# Show result
# ----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()