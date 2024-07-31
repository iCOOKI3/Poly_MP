import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "models/best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

def preprocess_image(image):
    # Convert image to RGB
    image = image.convert('RGB')
    return image

def infer_image(image):
    # Run inference
    results = model(image)
    print("Results type:", type(results))
    print("Results contents:", results)
    return results

def draw_predictions(image, results):
    # Convert PIL image to numpy array for drawing
    image_np = np.array(image)
    draw = ImageDraw.Draw(image)

    # Check if results is a list
    if isinstance(results, list):
        results = results[0]  # Get the first item if it's a list

    # Extract bounding boxes and labels
    boxes = results.boxes
    names = results.names
    
    for box in boxes:
        # Accessing box information
        box_xyxy = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()  # Confidence score
        cls = int(box.cls[0].cpu().numpy())  # Class index
        label = names[cls] if cls in names else 'Unknown'
        
        # Coordinates
        x1, y1, x2, y2 = box_xyxy
        
        # Draw bounding box and label
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1), f"{label} {conf:.2f}", fill='red')

    return image

st.title("Staff Face Recognition with YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = preprocess_image(image)
    
    # Run inference
    results = infer_image(image)
    
    # Draw predictions
    result_image = draw_predictions(image, results)
    
    st.image(result_image, caption='Processed Image', use_column_width=True)
