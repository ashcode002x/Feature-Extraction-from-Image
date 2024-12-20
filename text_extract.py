import cv2
import csv
import requests
import numpy as np
import tensorflow as tf
from doctr.models import ocr_predictor
from io import BytesIO

# Initialize doctr OCR predictor
ocr_model = ocr_predictor(pretrained=True)

# Function to download an image from a URL
def download_image(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to download image from {image_url}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Function to correct image rotation using TensorFlow
def correct_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if angle < -45:
                angle += 180
            elif angle > 45:
                angle -= 180
            angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            if median_angle > 45:
                median_angle -= 90
            if median_angle < -45:
                median_angle += 90
            
            # Rotate the image to correct the orientation using TensorFlow
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            angle_rad = tf.constant(median_angle * (np.pi / 180), dtype=tf.float32)
            rotated = tf.image.rot90(image_tensor, k=int(angle_rad))
            rotated = rotated.numpy().astype(np.uint8)
            return rotated
    return image

# Function to process an image and extract text
def extract_text_from_image(image):
    if image is None:
        print("Invalid image!")
        return None
    
    # Correct rotation if necessary
    image = correct_rotation(image)
    
    # Convert the image to RGB (since OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform OCR on the image
    result = ocr_model([image_rgb])  # Pass as a list

    # Extract the text
    text = result.render()
    return text

# Function to handle images with mixed orientations
def extract_text_from_mixed_orientations(image):
    # Placeholder for additional orientation processing
    text = extract_text_from_image(image)
    return text

# Read the CSV file and process each image link
with open('dataset/sample_test.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_url = row['image_link']
        entity_name = row['entity_name']

        # Download the image
        print(f"Processing image for entity: {entity_name} from {image_url}")
        image = download_image(image_url)

        # Extract text from the image
        text = extract_text_from_mixed_orientations(image)

        if text:
            print(f"Extracted Text for {entity_name}: {text}")
