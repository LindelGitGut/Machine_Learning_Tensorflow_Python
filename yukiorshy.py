import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Verzeichnis der Daten
data_dir = 'data/'

# Bildgrößen und Batchgröße
img_height, img_width = 32, 32
batch_size = 32
model_path = 'cat_classifier_model.keras'  # Verwenden des empfohlenen Formats

# Funktion zur Überprüfung und Entfernung leerer/beschädigter Bilder
def is_valid_image(file_path):
    try:
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=3)
        if tf.reduce_sum(img) > 0:
            return True
        else:
            return False
    except:
        return False

def clean_dataset(data_dir):
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if img_file.startswith("not_usable"):
                    continue
                if not is_valid_image(img_path):
                    new_img_path = os.path.join(class_path, "not_usable_" + img_file)
                    print(f"Ungültiges Bild erkannt und umbenannt: {img_path} -> {new_img_path}")
                    os.rename(img_path, new_img_path)

# Daten bereinigen
clean_dataset(data_dir)

# Dataset laden
def load_dataset(data_dir, validation_split, subset):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

try:
    train_ds = load_dataset(data_dir, validation_split=0.2, subset="training")
    val_ds = load_dataset(data_dir, validation_split=0.2, subset="validation")
except Exception as e:
    print(f"Fehler beim Laden des Datasets: {e}")
    exit(1)

# Automatische Erkennung der Klassen (Ordnernamen)
class_names = train_ds.class_names
print(f'Klassennamen: {class_names}')

# Normalisierungsschicht
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Bildaugmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch für Leistung
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Modell trainieren oder weitertrainieren
def train_model(continue_training=False):
    if continue_training and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Vorhandenes Modell geladen.")
    else:
        # Neues Modell erstellen
        model = tf.keras.models.Sequential([
            layers.Input(shape=(img_height, img_width, 3)),  # Input Layer hinzugefügt
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(class_names), activation='softmax')  # Anpassen auf 3 Outputs für Mehrklassen-Klassifikation
        ])

        # Modell kompilieren
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Modell trainieren
    try:
        model.fit(
            train_ds,
            epochs=100,  # Anzahl der Epochen erhöht auf 100
            validation_data=val_ds
        )
    except Exception as e:
        print(f"Fehler beim Training des Modells: {e}")
        exit(1)

    # Modell speichern
    model.save(model_path)
    print("Modell gespeichert.")
    return model

# Modell trainieren oder weitertrainieren
model = train_model(continue_training=True)

def classify_image(img_path, model):
    try:
        img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_class_idx]

        return predicted_class if predicted_class != "no_pet" else "No Pet"
    except Exception as e:
        print(f"Fehler bei der Bildklassifizierung: {e}")
        return None

# Beispiel für das Testen eines neuen Bildes
img_path = 'test.jpg'
predicted_class = classify_image(img_path, model)
if predicted_class is not None:
    print(f'Die vorhergesagte Klasse ist: {predicted_class}')
else:
    print("Konnte die Klasse des Bildes nicht vorhersagen.")
