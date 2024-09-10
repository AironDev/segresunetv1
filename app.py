import torch
import streamlit as st
from torchvision import transforms
from PIL import Image

# Load your pre-trained model
@st.cache_resource  # This ensures the model is cached
def load_model():
    model = torch.load("best_metric_model.pth", map_location=torch.device('cpu'))
    model.eval()  # Put the model in evaluation mode
    return model

model = load_model()

# Define a function to preprocess the input image
def preprocess_image(image):
    # Example preprocessing steps (adjust based on your model’s requirements)
    preprocess = transforms.Compose([
        transforms.Resize((240, 240)),  # Resize image if needed
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Define the prediction function
def predict(image_input):
    with torch.no_grad():
        prediction = model(image_input)
        return prediction

# Streamlit app
def main():
    st.title("Tumor Segmentation Model Deployment")
    st.write("Upload an MRI scan image for tumor segmentation.")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an MRI scan...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Preprocess the image
        image_input = preprocess_image(image)

        # Run prediction
        prediction = predict(image_input)

        # Display the original image
        st.image(image, caption='Uploaded MRI scan.', use_column_width=True)

        # Display prediction
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
