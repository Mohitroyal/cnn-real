import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

#loding data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(X_train.shape)
X_train_scaled = X_train/255
X_test_scaled = X_test/255
#onehot encoding
y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print(classes[y_train[3][0]])
# auggmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train_scaled)
# cnn model
cnn = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_scaled, y_train_categorical, epochs=10)
# Save the original model
cnn.save("model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
tflite_model = converter.convert()
# Save as .tflite file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
# Evaluate
cnn.evaluate(X_test_scaled, y_test_categorical)
# Predict
y_pred = cnn.predict(X_test_scaled)
print(np.argmax(y_pred[0]))
print(classes[np.argmax(y_pred[0])])