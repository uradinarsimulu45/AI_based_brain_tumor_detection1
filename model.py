"""
Brain Tumor Detection Model Training (MobileNetV2)
--------------------------------------------------
✅ Compatible with TensorFlow 2.16+ (Keras 3)
✅ Model size < 20 MB
✅ Works on macOS (Metal) + Windows/Linux
✅ Perfect for Streamlit Cloud
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

print("✅ TensorFlow version:", tf.__version__)
print("🔍 GPU Available:", tf.config.list_physical_devices('GPU'))

# macOS thread safety
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Dataset setup
data_dir = "dataset"
img_size = (224, 224)
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Base model: MobileNetV2 (pretrained on ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze layers

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=1
)

# Save model
model.save("brain_tumor_model.h5")
print("✅ Model trained and saved as brain_tumor_model.h5 (~15 MB)")
