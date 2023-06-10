import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import streamlit as st

# Model loader function (cached)
@st.cache_resource()  # Cache the model
def load_model():
    # Display the loading animation

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.jit.load('./model/mask_detection_model.pt', map_location=device)
    # model = None

    # Define the transformation to apply to the input image
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the class labels of your model
    class_labels = ['with_mask', 'without_mask']  # Replace with your actual class labels

    return model,transform,class_labels

# Function to make a prediction on the input image
def predict(image, transform, model, class_labels):
    # Apply the transformation to the input image
    image = transform(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Get the predicted label
    label = class_labels[predicted.item()]
    return label

# Create the Streamlit app
def main():
    st.title("Image Classification for Mask Detection")
    st.write("Upload an image and get a prediction whether the person in the image is wearing a mask or not.")

    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Load the model, transformation and class labels
    with st.spinner('Loading model...'):
        time.sleep(0.1)
        model,transform,class_labels = load_model()
    st.success('Model loaded successfully.')

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction
        label = predict(image, transform, model, class_labels)
        st.write(f"Predicted Label: {label}")

# Run the app
if __name__ == "__main__":
    main()
