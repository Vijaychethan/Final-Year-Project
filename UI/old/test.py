import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

def main():
    st.title("Paadaraksha")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Get the file path of the uploaded image
        file_path = get_file_path(uploaded_file)
        st.write(f"File Path: {file_path}")

        # Perform prediction and display the result
        if st.button("Run Prediction"):
            prediction(file_path)
            st.write("Button")

def get_file_path(uploaded_file):
    # Check if the uploaded file has a name attribute
    if hasattr(uploaded_file, 'name'):
        return uploaded_file.name
    else:
        return "File path not available."

def prediction(file_path):
    # Define the path to your YOLOv8 instance segmentation model
    model_path = "yolov8m-seg.pt"

    # Load the YOLOv8 instance segmentation model
    model = YOLO(model_path)

    predict=model.predict("assets/"+file_path,save=True,save_txt=True)
    st.write(predict[0])

if __name__ == "__main__":
    main()
