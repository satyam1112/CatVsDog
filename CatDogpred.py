import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the trained model
model = load_model("CatVsDogModel50.h5")

# Create a Streamlit app
st.title("🐱🐶 Cat vs. Dog Classifier 🐱🐶")
st.sidebar.markdown(
    """
    <div style="text-align:center; padding-top: 80px;">
        <h1>Cat vs. Dog Classifier</h1>
        <p>Made by Satyam</p>
    </div>
    """,
    unsafe_allow_html=True
)
# Initialize the image variable
image = None

# Upload an image or paste a URL
option = st.radio("Choose an option:", ("Upload an image", "Paste an image URL"))

if option == "Upload an image":
    try:
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image)
    except :
        st.error("Invalid Image")
elif option == "Paste an image URL":
    url = st.text_input("Paste the image URL:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Error downloading the image. Please check the URL.")
            image = None

# Function to make predictions
def make_prediction(image):
    if image is not None:
        try:
            # Convert the image to a format suitable for prediction
            img = np.array(image)
            img = cv2.resize(img, (128, 128))
            img_inp = img.reshape((1, 128, 128, 3))
            
            # Make a prediction
            predictions = model.predict(img_inp)
            
            # Determine the class label
            if predictions[0][0] >= 0.5:
                return "Dog"
            else:
                return "Cat"
        except ValueError:
            st.error("Error processing the image. Please ensure it is a valid image.")
            return None
    else:
        return None

# Display the prediction
if image is not None:
    st.image(image, caption="Uploaded Image or Image from URL", use_column_width=True)
    prediction = make_prediction(image)
    if prediction:
        print(f"Prediction: It's a {prediction}!")
        st.subheader(f"Prediction: It's a {prediction}!")
else:
    st.write("Please upload an image or paste an image URL.")
