import torch
import streamlit as st
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from model import HybridSegResNetUNet  # Import the model from the model.py file

# Load the saved model
@st.cache_resource
def load_model():
    model = HybridSegResNetUNet(in_channels=5, out_channels=3)
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define a function to preprocess the input image
def preprocess_image(image):
    # Add your preprocessing steps here (resizing, normalization, etc.)
    return image

# Define the inference function
def inference(image_input):
    # Assuming the input image is already preprocessed and in the right shape (torch.Tensor)
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image_input,
            roi_size=(240, 240, 160), 
            sw_batch_size=1, 
            predictor=model, 
            overlap=0.5
        )
    return output

# Streamlit app layout
st.title("HybridSegResNetUNet Medical Image Segmentation")
uploaded_file = st.file_uploader("Choose an image file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    input_image = preprocess_image(uploaded_file)  # You need to implement this function
    st.write("Running inference...")
    
    # Run inference
    output = inference(input_image)
    st.write("Inference completed.")

    # Post-process and display the output
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    processed_output = post_trans(output)
    st.write("Processed Output:", processed_output)
