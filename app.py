import streamlit as st
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR
from keras.models import load_model
import tempfile
from datetime import datetime  # Import the datetime module

# Load the Keras model for Fruit Vision
fruit_model = load_model('fruit_classifier_model.h5')

# Class mapping for Fruit Vision
fruit_class_mapping = {
    0: "Fresh Apples",
    1: "Fresh Bananas",
    2: "Fresh Oranges",
    3: "Rotten Apples",
    4: "Rotten Bananas",
    5: "Rotten Oranges"
}

def preprocess_image(image):
    """Preprocess image for fruit classification."""
    image = image.convert('RGB')
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image):
    """Predict the fruit type and freshness."""
    processed_image = preprocess_image(image)
    prediction = fruit_model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return fruit_class_mapping[predicted_class_index]

# Initialize PaddleOCR for Smart Scan Product
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_brand_weight(image_np, brands):
    """Extract brand and weight information from an image."""
    ocr_results = ocr.ocr(image_np, cls=True)
    results = []
    
    for detection in ocr_results[0]:
        box, (text, confidence) = detection
        if confidence > 0.8:  # Confidence threshold
            # Check for brand name
            brand_pattern = re.compile(r'|'.join(brands), re.IGNORECASE)
            brand_match = brand_pattern.search(text)
            brand_name = brand_match.group(0) if brand_match else 'Brand not found'

            # Check for net weight
            weight_match = re.search(r'NET (?:WEIGHT|WT|QUANTITY|Wt|g|ml):?\s*([\d.]+)\s*(kg|g)', text, re.IGNORECASE)
            net_weight = f"{weight_match.group(1)} {weight_match.group(2)}" if weight_match else 'Net Weight not found'
            
            results.append({"Brand Name": brand_name, "Net Weight": net_weight, "Text": text})
    
    return results

# Streamlit App
st.title("Integrated App")
st.sidebar.title("Model Selection")
models = ["Multiple Product Smart Scan", "Single Product Smart Scan", "Fruit Vision"]
selected_model = st.sidebar.selectbox("Select a Model:", models)

# Common image capture/upload options using st.camera_input
def capture_image(option):
    """Capture image either via webcam or file upload"""
    if option == "Capture from Webcam":
        # Webcam capture using Streamlit's camera input
        uploaded_file = st.camera_input("Capture an image")

        if uploaded_file is not None:
            # Save the captured image to a temporary file
            image = Image.open(uploaded_file)
            st.image(image, caption='Captured Image', use_column_width=True)
            return image
        else:
            st.write("No image captured.")
            return None

    else:
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            return image
        return None

# Function to save results to a text file with date and time
def save_results_to_file(results_str):
    """Save results string to a text file with date and time."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current date and time
    results_with_date = f"Results generated on: {current_time}\n\n{results_str}"  # Append the date and time to the results
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(results_with_date.encode())
        tmpfile.close()
        return tmpfile.name

if selected_model == "Multiple Product Smart Scan":
    st.write("**Multiple Product Smart Scan**")
    option = st.radio("Choose Input Method:", ["Upload an Image", "Capture from Webcam"])
    
    image = capture_image(option)
    
    if image:
        image_np = np.array(image)
        
        # Define the list of brand names
        brands = ["Bikaji", "VADAI","Good Day", "Haldiram", "Dettol", "Fortune", "BRU", "BEARDO", "PERK Mini Treats", "MAGGI", "KitKat", "Kissan", "Del Monte", "Bourn"]
        
        # Extract brand and weight
        results = extract_brand_weight(image_np, brands)
        
        result_str = ""
        if results:
            st.write("**Extracted Information:**")
            for i, result in enumerate(results):
                st.write(f"**Product {i+1}:**")
                st.write(f"- Brand Name: {result['Brand Name']}")
                st.write(f"- Net Weight: {result['Net Weight']}")
                st.write(f"- Text: {result['Text']}")
                
                result_str += f"Product {i+1}:\n"
                result_str += f"- Brand Name: {result['Brand Name']}\n"
                result_str += f"- Net Weight: {result['Net Weight']}\n"
                result_str += f"- Text: {result['Text']}\n\n"
        else:
            st.write("No relevant information found in the image.")
            result_str = "No relevant information found in the image."
        
        # Save to text file
        file_path = save_results_to_file(result_str)
        st.download_button("Download Results as Text File", data=open(file_path, "rb"), file_name="results.txt")

elif selected_model == "Single Product Smart Scan":
    st.write("**Single Product Smart Scan**")
    option = st.radio("Choose Input Method:", ["Upload an Image", "Capture from Webcam"])
    
    image = capture_image(option)
    
    if image:
        # If an image is captured or uploaded, process it
        image = image.convert('RGB')
        image_np = np.array(image)
        
        # Use OCR to scan the image for brand and weight
        ocr_result = ocr.ocr(image_np, cls=True)
        combined_text = " ".join([line[1][0] for line in ocr_result[0]]) if ocr_result else ""
        
        # Define the list of brand names
        brands = ["Bikaji", "Good Day", "Haldiram", "MAGGI", "KitKat", "Kissan", "Del Monte"]
        
        # Extract brand and weight information using regex
        brand_pattern = re.compile(r'|'.join(brands), re.IGNORECASE)
        brand_match = brand_pattern.search(combined_text)
        brand_name = brand_match.group(0) if brand_match else 'Brand not found'
        
        weight_match = re.search(r'NET (?:WEIGHT|WT|QUANTITY|Wt):?\s*([\d.]+)\s*(kg|g)', combined_text, re.IGNORECASE)
        net_weight = f"{weight_match.group(1)} {weight_match.group(2)}" if weight_match else 'Net Weight not found'
        
        # Display results
        st.write(f"Brand Name: {brand_name}")
        st.write(f"Net Weight: {net_weight}")
        
        result_str = f"Brand Name: {brand_name}\nNet Weight: {net_weight}\n"
        
        # Save to text file
        file_path = save_results_to_file(result_str)
        st.download_button("Download Results as Text File", data=open(file_path, "rb"), file_name="results.txt")

elif selected_model == "Fruit Vision":
    st.write("**Fruit Vision**")
    option = st.radio("Choose Input Method:", ["Upload an Image", "Capture from Webcam"])
    
    image = capture_image(option)
    
    if image:
        # Predict fruit type
        predicted_class = predict_image(image)
        st.write(f'The Fruit is: {predicted_class}')
        
        result_str = f"The Fruit is: {predicted_class}\n"
        
        # Save to text file
        file_path = save_results_to_file(result_str)
        st.download_button("Download Results as Text File", data=open(file_path, "rb"), file_name="results.txt")
