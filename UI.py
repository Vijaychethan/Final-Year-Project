import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch



def main():
    st.set_page_config(layout="wide")
   
    bg_img = '''
    <style>
            [data-testid="stAppViewContainer"] {
            background-image: url('https://media.gettyimages.com/id/1307928215/photo/footprints-with-mannequin-feet-podiatry-concept.jpg?s=612x612&w=0&k=20&c=p72CjWtcEHmYsrH6sar1oK3DEzRDQaDJfglzGoO8GEw=');
            background-position : center;
            background-size: cover;
            background-repeat: no-repeat;
            }
    </style>
    '''
    st.markdown(bg_img, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.05,4,3.95])

    with col2:
        
        st.markdown("<h1>Paadaraksha</h1>", unsafe_allow_html=True)
        st.write("")
        # st.write("This text is left-aligned by default, within the middle column.")
        


        selected_gender = st.selectbox("Select Your Gender",["Male","Female"])
        # selected_option = st.selectbox("Select Your Sole size", ["4", "5", "6","7","8","9","10"])
        options_to_images_M = {
            "5": "BluePrints\Men\M5.png",
            "6": "BluePrints\Men\M6.png",
            "7": "BluePrints\Men\M7.png",
            "8": "BluePrints\Men\M8.png",
            "9": "BluePrints\Men\M9.png",
            "10": "BluePrints\Men\M10.png",
            "11": "BluePrints\Men\M11.png"
        }
        options_to_images_F = {
            "4": "BluePrints\Women\W4.png",
            "5": "BluePrints\Women\W5.png",
            "6": "BluePrints\Women\W6.png",
            "7": "BluePrints\Women\W7.png",
            "8": "BluePrints\Women\W8.png",
            "9": "BluePrints\Women\W9.png",
            "10": "BluePrints\Women\W10.png"
        }

        if selected_gender =="Male":
            selected_option = st.selectbox("Select Your Sole size", [ "5", "6","7","8","9","10","11"])
            image_path = options_to_images_M.get(selected_option, None)
        if selected_gender =="Female":
            selected_option = st.selectbox("Select Your Sole size", ["4", "5", "6","7","8","9","10"])
            image_path = options_to_images_F.get(selected_option, None)
        # Get the image path based on the selected option
        

    

        # Add a button to start scanning
        if st.button("Scan"):
            # Load blueprint image
            blueprint_img = cv2.imread(image_path)
            blueprint_img = cv2.cvtColor(blueprint_img, cv2.COLOR_BGR2RGB)
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

def get_mask(results, class_id, color):
    masks = results[0].masks.data
    indices = torch.where(results[0].boxes.data[:, 5] == class_id)
    class_masks = masks[indices]
    class_mask = torch.any(class_masks, dim=0).int() * 255
    class_mask_numpy = class_mask.cpu().numpy()

    # Ensure the mask has a valid depth for color conversion
    class_mask_rgb = cv2.cvtColor(class_mask_numpy.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Create a binary mask for non-zero values
    binary_mask = class_mask_numpy != 0

    # Apply color assignment separately for each channel
    for i in range(3):  # Loop over channels (R, G, B)
        class_mask_rgb[:, :, i][binary_mask] = color[i]

    return class_mask_rgb.astype(np.uint8)


def Predict(pil_image, class_name):
    model_path = "Models/best.pt"

    # Load the YOLOv8 instance segmentation model
    model = YOLO(model_path)

    # Make predictions on the input image
    # model.predict(source=pil_image.copy(),show=True,show_labels=True, save=True, save_txt=False)

    person_results = list(model.predict(source=pil_image, save=True, save_txt=False, stream=True))
    person_mask = get_mask(person_results, class_id=1, color=(255, 255, 255))
    return person_mask


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

    # Segment a particular class (e.g., "person")
    class_name = "Ulcers"  # Specify the class you want to segment
    segmented_image = Predict(pil_image, class_name)
    result=Image.fromarray(segmented_image, mode='RGB')
    dest_xor = cv2.bitwise_xor(blueprint_img, segmented_image, mask = None)
    # Display the segmented image using Streamlit
    st.image(dest_xor, caption=f'Segmented ', use_column_width=True)




if __name__ == "__main__":
    main()


# done till foot overlapping and masked output
