import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Plant_disease_model.h5')

# Class names
CLASS_NAMES = ('Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight')

# Set the title of the app
st.title('Potato Leaf Disease Detection')
st.markdown('Upload an image of the potato leaf')

# Upload image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Display the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resize the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 dimensions
        opencv_image = np.reshape(opencv_image, (1, 256, 256, 3))
        
        # Make prediction
        y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(y_pred)]
        
        # Display the prediction result
        st.write('Prediction:', result)

        # Display the probability scores for each class
        st.write('Confidence Scores:')
        for class_name, score in zip(CLASS_NAMES, y_pred[0]):
            st.write(f'{class_name}: {score:.2%}')
