import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    # Set title
    st.title("Ulcer Masking")

    # Main content
    st.write("Welcome to Ulcer Masking Web App!")

    # Load blueprint image
    blueprint_img = cv2.imread("masktrail.png")
    blueprint_img = cv2.cvtColor(blueprint_img, cv2.COLOR_BGR2RGB)

    # Add a button to start scanning
    if st.button("Scan"):
        # Function to open camera window
        open_camera(blueprint_img)

def open_camera(blueprint_img):
    # OpenCV code to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    # Read frames from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize blueprint image to fit camera frame while preserving aspect ratio
        scale_factor = min(frame.shape[0] / blueprint_img.shape[0], frame.shape[1] / blueprint_img.shape[1])
        resized_blueprint = cv2.resize(blueprint_img, None, fx=scale_factor, fy=scale_factor)

        # Calculate the position to center the blueprint image
        top = (frame.shape[0] - resized_blueprint.shape[0]) // 2
        left = (frame.shape[1] - resized_blueprint.shape[1]) // 2

        # Create an empty overlay frame and place the resized blueprint image onto it
        overlay = np.zeros_like(frame)
        overlay[top:top + resized_blueprint.shape[0], left:left + resized_blueprint.shape[1]] = resized_blueprint

        # Blend the overlay with the camera frame
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display the frame
        cv2.imshow("Camera", frame)

        # Check for key press to exit or capture image
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):  # Press 'c' to capture image
            capture_image(frame,blueprint_img)

    # Release the camera and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()

def capture_image(frame, blueprint_img):
    # Convert blueprint image to grayscale and threshold to binary mask
    blueprint_gray = cv2.cvtColor(blueprint_img, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(blueprint_gray, 1, 255, cv2.THRESH_BINARY)

    # Resize mask to match frame size
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Apply bitwise_and operation
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # Display the captured image using Streamlit
    st.image(pil_image, caption='Captured Image', use_column_width=True)


if __name__ == "__main__":
    main()


# masking out input works