import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# --------------------
# CONFIG
# --------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# --------------------
# DATA GENERATORS
# --------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# --------------------
# MODEL
# --------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Unfreeze last 50 layers for fine-tuning
for layer in base_model.layers[:-50]:
    layer.trainable = False

for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# --------------------
# COMPILE
# --------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# --------------------
# CLASS WEIGHTS
# --------------------
class_weights = {0: 1.0, 1: 1.2}  # slight preference for melanoma

# --------------------
# CALLBACKS
# --------------------
lr_callback = ReduceLROnPlateau(
    monitor='val_auc', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

# --------------------
# TRAIN
# --------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[lr_callback]
)

# --------------------
# EVALUATE
# --------------------
pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.3).astype(int)  # lower threshold for early warning

print("\nCLASSIFICATION REPORT:")
print(classification_report(test_data.classes, pred_labels))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(test_data.classes, pred_labels))

# --------------------
# SAVE MODEL
# --------------------
model.save("skin_cancer_model.keras")
print("\nModel saved as skin_cancer_model.keras")