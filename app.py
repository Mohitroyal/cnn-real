import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🚀 CIFAR-10 Image Classifier (TFLite)")
st.markdown("Upload a 32x32 image to classify it!")
uploaded_file = st.file_uploader("Choose a 32x32 image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((32, 32))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    st.subheader(f"Prediction: `{classes[predicted_class]}`")
    st.write(f"Confidence: `{confidence:.2f}%`")

