from flask import Flask, render_template, request, send_file
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Load the YOLOv8 instance segmentation model
model_path = "yolov8m-seg.pt"
model = YOLO(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']

        if file:
            # Read the image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Perform instance segmentation for person
            person_results = list(model.predict(source=img, save=True, save_txt=False, stream=True))
            person_mask = get_mask(person_results, class_id=0, color=(0, 0, 255))  # Red color for person

            # Map the person mask to another image (background)
            background_image = cv2.imread('assets\ppltest.jpg')  # Replace with the path to your background image
            mapped_result = map_mask_to_image(person_mask, background_image)

            # Save the combined and mapped result to a BytesIO object
            output_buffer = BytesIO()
            Image.fromarray(mapped_result, mode='RGB').save(output_buffer, format='JPEG')
            output_buffer.seek(0)

            # Return the combined and mapped result to the user
            return send_file(output_buffer, mimetype='image/jpeg')

    return render_template('index.html')

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

def map_mask_to_image(mask, background_image):
    # Resize the mask to match the background image dimensions
    resized_mask = cv2.resize(mask, (background_image.shape[1], background_image.shape[0]))

    # Map the resized mask to the background image
    mapped_result = cv2.addWeighted(resized_mask, 1, background_image, 1, 0)

    return mapped_result

if __name__ == '__main__':
    app.run(debug=True)
