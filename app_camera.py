import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime

# Define labels
class_labels = {
    0: "Defective",
    1: "Not Defective"
}

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("Bottle Defect Detection")
st.write("Use your laptop camera to detect bottle defects in real-time")

# Camera input section
st.header("📸 Camera Capture")
img_file_buffer = st.camera_input("Take a picture of the bottle")

# Alternative file upload (fallback)
st.header("📁 Alternative: File Upload")
uploaded_file = st.file_uploader("Or upload an image of a bottle", type=["jpg", "jpeg", "png", "webp"])

# Process camera input
if img_file_buffer is not None:
    st.divider()
    st.subheader("Camera Analysis")
    
    # Convert camera input to PIL Image
    image = Image.open(img_file_buffer).convert("RGB")
    
    # Display the captured image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Captured Image", use_column_width=True)
    
    with col2:
        with st.spinner("Analyzing bottle..."):
            # Resize and preprocess image
            size = (224, 224)
            processed_image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            image_array = np.asarray(processed_image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            
            predicted_label = np.argmax(predictions)
            confidence = predictions[predicted_label]
            
            # Display results
            st.write("**Analysis Results:**")
            if predicted_label == 0:  # Defective
                st.error(f"⚠️ **DEFECTIVE**")
                st.metric("Confidence", f"{confidence:.2%}")
                st.write("The bottle appears to have defects!")
            else:  # Not Defective
                st.success(f"✅ **NOT DEFECTIVE**")
                st.metric("Confidence", f"{confidence:.2%}")
                st.write("The bottle appears to be in good condition!")
            
            if st.checkbox("Show raw scores"):
                st.write("Raw prediction scores:", predictions)

# Process file upload (fallback)
elif uploaded_file is not None:
    st.divider()
    st.subheader("File Analysis")
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing bottle..."):
        # Resize and preprocess image
        size = (224, 224)
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_label = np.argmax(predictions)
        confidence = predictions[predicted_label]
        
        # Display results
        st.write("**Analysis Results:**")
        if predicted_label == 0:  # Defective
            st.error(f"⚠️ **DEFECTIVE** (Confidence: {confidence:.2f})")
        else:  # Not Defective
            st.success(f"✅ **NOT DEFECTIVE** (Confidence: {confidence:.2f})")
        
        if st.checkbox("Show raw scores"):
            st.write("Raw prediction scores:", predictions)

# Instructions
with st.expander("📋 How to Use"):
    st.write("""
    **Using Camera Mode:**
    1. Allow camera access when prompted by your browser
    2. Position your bottle in front of the laptop camera
    3. Ensure good lighting conditions
    4. Click "Take Photo" when ready
    5. The app will automatically analyze the image
    
    **Tips for Best Results:**
    - Hold the bottle steady
    - Fill most of the camera frame with the bottle
    - Avoid shadows and reflections
    - Ensure the bottle is clearly visible
    
    **Fallback Option:**
    - If camera doesn't work, use the file upload option
    """)

# Model information
with st.expander("ℹ️ Model Information"):
    st.write("""
    **Model Details:**
    - Model: TFLite quantized model
    - Input size: 224x224 pixels
    - Classes: Defective (0), Not Defective (1)
    - Framework: TensorFlow Lite
    """)
