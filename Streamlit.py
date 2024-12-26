import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model (ensure that the model is saved in the same directory or provide the correct path)
model = tf.keras.models.load_model("classifer.keras")

# Streamlit app
st.title("Digit Recognition App")
st.write("Draw a digit on the canvas below (28x28 pixels).")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Black background
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Display the canvas image
    # st.image(canvas_result.image_data, width=280)

    # Convert the image to grayscale and resize to 28x28
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img / 255.0

    # Predict the digit
    img = img.reshape(1, 28, 28, 1)
    predictions = model.predict(img)

    # Filter predictions with probabilities greater than 0.7
    threshold = 0.5
    filtered_indices = np.where(predictions[0] > threshold)[0]

    st.write("Predicted digits with probability > 0.5:")
    for idx in filtered_indices:
        st.write(f"Digit: {idx}, Probability: {predictions[0][idx]:.4f}")
