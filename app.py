import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("Age & Gender Prediction Dashboard")
st.write("Upload an image, and our multi-task ResNet model will predict both traits.")


# Load the model once and cache it so it doesn't reload on every user click
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("/Users/abhigyantripathi/Desktop/ml_practice_projects/non-liner-model/age_gender_predictor.keras")


model = load_my_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Analyzing image...")

    # 2. Replicate your exact training preprocessing pipeline using TensorFlow
    # Convert the uploaded file bytes into a tensor
    file_bytes = uploaded_file.getvalue()
    image_tensor = tf.io.decode_jpeg(file_bytes, channels=3)

    # Resize exactly to [200, 200] matching your training setup
    resized_image = tf.image.resize(image_tensor, [200, 200])

    # Convert to numpy array and add the batch dimension required by the model
    img_array = resized_image.numpy()
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Run Inference
    age_preds, gender_preds = model.predict(img_array)

    # Process predictions
    predicted_age = int(np.round(age_preds[0][0]))
    gender_prob = gender_preds[0][0]
    predicted_gender = "Male" if gender_prob > 0.5 else "Female"

    # Layout results elegantly in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Gender", value=predicted_gender)
    with col2:
        st.metric(label="Predicted Age", value=f"{predicted_age} years old")