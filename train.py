import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

train_dataset_path = 'path\train'
test_dataset_path = 'path\test'
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

def create_model():
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)


model.save('license_plate_detector.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (128, 128))
    resized_image = resized_image.reshape(1, 128, 128, 1) / 255.0
    return resized_image

def predict_license_plate(image_path, model):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction > 0.5  


if __name__ == "__main__":
    model = tf.keras.models.load_model('license_plate_detector.h5')
    test_image_path = 'path\test\license_plate\test_image.png'
    is_license_plate = predict_license_plate(test_image_path, model)
    print(f"Номерной знак {'найден' if is_license_plate else 'не найден'}")
