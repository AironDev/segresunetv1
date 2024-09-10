import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import nibabel as nib
import numpy as np
from monai.transforms import Compose, Resize, ScaleIntensity, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# Define your HybridSegResNetUNet model class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HybridSegResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters=16, dropout_prob=0.2):
        super(HybridSegResNetUNet, self).__init__()
        
        self.encoder1 = ResidualBlock(in_channels, init_filters)
        self.encoder2 = ResidualBlock(init_filters, init_filters * 2, stride=2)
        self.encoder3 = ResidualBlock(init_filters * 2, init_filters * 4, stride=2)
        self.encoder4 = ResidualBlock(init_filters * 4, init_filters * 8, stride=2)
        
        self.bottleneck = ResidualBlock(init_filters * 8, init_filters * 16, stride=2)
        
        self.decoder4 = nn.ConvTranspose3d(init_filters * 16, init_filters * 8, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose3d(init_filters * 8, init_filters * 4, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose3d(init_filters * 4, init_filters * 2, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose3d(init_filters * 2, init_filters, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.decoder4(b)
        d4 = d4 + e4
        d4 = F.relu(d4)
        
        d3 = self.decoder3(d4)
        d3 = d3 + e3
        d3 = F.relu(d3)
        
        d2 = self.decoder2(d3)
        d2 = d2 + e2
        d2 = F.relu(d2)
        
        d1 = self.decoder1(d2)
        d1 = d1 + e1
        d1 = F.relu(d1)
        
        out = self.final_conv(d1)
        return out

# GPU fallback handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
@st.cache_resource
def load_model():
    model = HybridSegResNetUNet(in_channels=5, out_channels=3).to(device)
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Preprocess NIfTI medical image
def preprocess_image(image_file):
    # Check if the file is .nii or .nii.gz
    if image_file.endswith(".gz"):
        img = nib.load(image_file)  # Load compressed NIfTI
    else:
        img = nib.load(image_file)  # Load uncompressed NIfTI
    
    # Get image data as a NumPy array
    img_data = img.get_fdata()
    
    # Add a batch and channel dimension
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension (since in_channels=5)

    # Apply transformations (resizing, intensity scaling, etc.)
    transforms = Compose([
        Resize(spatial_size=(240, 240, 160)),  # Resize to match the expected input shape
        ScaleIntensity(minv=0.0, maxv=1.0)  # Normalize the pixel intensity
    ])
    
    # Apply transformations
    img_data = transforms(img_data)

    # Convert to torch.Tensor
    img_tensor = torch.Tensor(img_data).to(device)

    return img_tensor

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
uploaded_file = st.file_uploader("Choose a NIfTI image file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_file.nii.gz" if uploaded_file.name.endswith(".gz") else "temp_file.nii"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image file
    input_image = preprocess_image(temp_file_path)
    st.write("Running inference...")

    # Run inference
    output = inference(input_image)
    st.write("Inference completed.")
    
    # Post-process and display the output
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    processed_output = post_trans(output)

    st.write("Processed Output:", processed_output)
