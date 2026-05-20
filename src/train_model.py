import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path

# --- CONFIGURATION ---
# Absolute path to prevent "0 files found" errors
DATA_DIR = Path(r"E:\Projects\Fructoxis\data\Fructoxis_Dataset")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)


def train_unified_model():
    print("1. Loading Massive 12-Class Dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\nDiscovered {len(class_names)} Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n2. Assembling Hyper-Accurate Architecture...")
    # A. Upgraded Data Augmentation to handle YOLO cropped images better
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),  # Spin the fruits a bit more
        layers.RandomZoom(0.1),  # Zoom in/out slightly to simulate different camera distances
    ], name="augmentation_layer")

    # B. The GPU Upgrade: Fine-Tuning
    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    # Unfreeze the base model so we can alter its internal shape-recognition...
    base_model.trainable = True

    # ...but freeze the first 100 layers so we don't destroy its basic edge-detection skills
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # C. Build the Graph
    inputs = keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)

    # Bake the preprocessing directly into the network!
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=True)  # Set to True because we are fine-tuning now
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # 30% dropout prevents the model from memorizing the data
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # Use a tiny learning rate (0.00005) so we carefully adjust the top layers without breaking them
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00005),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print("\n3. Training Deep Fine-Tuned Model (20 Epochs)...")
    # This will run for 20 Epochs to get maximum accuracy for your presentation
    model.fit(train_ds, validation_data=val_ds, epochs=20)

    # Save to the root folder!
    model.save('fructoxis_v2.keras')
    print("\n[SUCCESS] Hyper-Accurate Model saved as 'fructoxis_v2.keras' in the root folder!")


if __name__ == "__main__":
    train_unified_model()