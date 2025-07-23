import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input # type: ignore
import numpy as np
from PIL import Image
import json
import os
from streamlit_cropper import st_cropper 

# Configuration 
MODEL_PATH = "Models/best_dog_breed_model.keras"
LABELS_PATH = "Labels/id_to_breed.json"
TARGET_SIZE = (224, 224)

# Page Configuration
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load Model and Labels
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Error: Labels file not found at {LABELS_PATH}.")
        return None
    with open(LABELS_PATH, 'r') as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return labels

model = load_model()
id_to_breed = load_labels()

if model is None or id_to_breed is None:
    st.stop()

# Image Preprocessing Function
def preprocess_image(img):
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess_input(img_array)
    return img_array


# Streamlit UI
st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog and crop it making sure the dog covers 90% of the image for best results")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

cropped_img = None

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    st.subheader("Crop the image to focus on the dog:")
    cropped_img = st_cropper(img, realtime_update=True, box_color="#0000FF",
                              aspect_ratio=None, return_type="image",
                              key="dog_cropper")

    if cropped_img:
        st.image(cropped_img, caption='Cropped Image (will be used for prediction)', use_column_width=True)
        st.write("")
        st.write("Classifying cropped image...")

        processed_img = preprocess_image(cropped_img)
        predictions = model.predict(processed_img)

        predicted_class_id = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        if predicted_class_id in id_to_breed:
            predicted_breed = id_to_breed[predicted_class_id].replace('_', ' ').title()
            st.success(f"**Predicted Breed:** {predicted_breed}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            st.subheader("Top 5 Predictions:")
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            for i, idx in enumerate(top_5_indices):
                breed_name = id_to_breed[idx].replace('_', ' ').title()
                prob = predictions[0][idx] * 100
                st.write(f"{i+1}. {breed_name}: {prob:.2f}%")
        else:
            st.error("Could not find breed name for the predicted class ID.")
            st.write("Raw prediction probabilities:", predictions[0])
    else:
        st.info("Please drag a box over the dog in the image above and click outside the box to confirm your crop.")
else:
    st.info("Please upload an image to begin.")

st.markdown("""*Disclaimer: This model is for demonstration purposes and may be inaccurate at times*""")