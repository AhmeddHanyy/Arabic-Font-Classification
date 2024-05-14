import streamlit as st
import joblib
import numpy as np
from PIL import Image
import cv2
from model import *

# Load the trained model
try:
    model = joblib.load("pipeline.pkl")
except Exception as e:
    st.error("Error loading model: " + str(e))

# Define the Streamlit app
def main():
    st.title("Font Classification")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert the image to numpy array
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Convert the image to OpenCV object in RGB format
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        mapper = {0 : "IBM", 1: "Lemonada", 2: "Marhey", 3: "Scheherazade New"}
        # Make prediction
        prediction = int(model.predict([image_cv2])[0])

        # Display prediction
        st.success("Prediction: " + mapper[prediction])

# Run the app
if __name__ == "__main__":
    main()
