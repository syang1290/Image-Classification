from pathlib import Path
import tensorflow as tf
from keras import layers, Sequential
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------------------------
# Paths
data_dir = Path.home() / "Downloads/asl_alphabet_train"
saved_model_path = Path("asl_model.h5")
test_image_path = Path.home() / "Downloads/asl_alphabet_test/A_test.jpg"

# --------------------------
# Model parameters
batch_size = 32
img_height = 180
img_width = 180

# --------------------------
# Load class names from training directory
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
num_classes = len(class_names)
print("Classes:", class_names)

# --------------------------
# Check if saved model exists
if saved_model_path.exists():
    print("Loading saved model...")
    model = load_model(saved_model_path)
else:
    print("Training new model...")

    # Load training and validation datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Normalize images
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Data augmentation (avoid horizontal flip for ASL)
    data_augmentation = Sequential([
        layers.RandomRotation(0.1, input_shape=(img_height, img_width, 3)),
        layers.RandomZoom(0.1),
    ])

    # Build CNN model
    model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # Train model
    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Save the trained model
    model.save(saved_model_path)
    print(f"Model saved to {saved_model_path}")

# --------------------------
# Test a single image
img = tf.keras.utils.load_img(test_image_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) / 255.0

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(
    f"This image most likely belongs to {predicted_class} with a {confidence:.2f}% confidence."
)
