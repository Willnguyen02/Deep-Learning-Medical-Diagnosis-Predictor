# import libraries and modules
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Page configurations
st.set_page_config(
    page_title="Chest X-Ray Diagnosis",
    layout="wide",
)

# Set labels
class_labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
data_folder = os.path.join(current_dir, "data", "models") # get data folder

# Get paths to models
model_paths = {
    "ResNet18":    os.path.join(data_folder, "resnet18_best.pt"),
    "DenseNet121": os.path.join(data_folder, "densenet121_best.pt"),
    "CNN":         os.path.join(data_folder, "cnn_best.pt"),
    "MLP":         os.path.join(data_folder, "mlp_best.pt"),
}

# Set threshold to o.5 (any predicted probability below 0.5 is treated as False)
threshold = 0.5


# Define Models

# Custom MLP Model 
class MLP(nn.Module):
    def __init__(self, in_features=4, hidden1=64, hidden2=32, out_features=15):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden1)
        self.fc2 = nn.Linear(in_features=hidden1, out_features=hidden2)
        self.output = nn.Linear(in_features=hidden2, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# Custom CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3) # 64 x 126 x 126
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # 64 x 63 x 63

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3) # 32 x 61 x 61
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # 32 x 30 x 30

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3) # 16 x 28 x 28
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2) # 16 x 14 x 14

        self.fc1 = nn.Linear(in_features=16 * 14 * 14, out_features=32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=32, out_features=15)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# loading models
@st.cache_resource(show_spinner="Loading model weights…") # caches model results for better performacne
def load_model(model_name: str):

    path = model_paths[model_name] # select path to model based on the inputed model_name
    if not os.path.exists(path): # check if file exists
        return None

    device = torch.device("cpu") # select cpu
    try:
        state = torch.load(path, map_location=device, weights_only=True) # loads saved model weights from model_paths[model_name]

    except Exception:
        state = torch.load(path, map_location=device)

    # Train model using best weights from training
    # ResNet18 model
    if model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 15))

    # DenseNet121 model
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 15))

    # Custom CNN model (grayscale, 1-channel)
    elif model_name == "CNN":
        model = CNN()

    # MLP model (tabular patient metadata)
    elif model_name == "MLP":
        model = MLP()

    model.load_state_dict(state) # load best weights
    model.eval() # set model to eval/ inference mode
    return model, None


# preprocessing
data_preprocessing = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # set output channels to 1 (black and white)
    transforms.Resize((128, 128)),
    transforms.ToTensor(), # resize and convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]), # normalize image
])

# create a function that preprocessis PIL images and returns a model-ready tensor
def preprocess_image(pil_img: Image.Image, model_name: str) -> torch.Tensor:
    tensor = data_preprocessing(pil_img)        # shape - (1, 128, 128)

    if model_name in ("ResNet18", "DenseNet121"):
        tensor = tensor.repeat(3, 1, 1)         # shape - (3, 128, 128)

    return tensor.unsqueeze(0)                  # shape - (1, C, 128, 128)


# prediction function
def predict(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:

    # disable gradient tracking and pass image through model
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy() # convert logits to probabilities (0-1) using the sigmoid function
    return probs


# display results using streamlit
def show_results(probs: np.ndarray, model_name: str):

    # create a dataframe to hold all results
    df = pd.DataFrame({
        "Condition":   class_labels,
        "Probability": probs,
        "Detected":    probs >= threshold, # only detect if probability is greater then threshold
    }).sort_values("Probability", ascending=False).reset_index(drop=True) # sort probability from hightst to lowest

    # extract detected conditions and convert to a list
    detected_conditions = df[df["Detected"]]["Condition"].tolist()

    # create diagnosisi section
    st.subheader("Diagnosis")
    if detected_conditions:
        st.error("**Findings detected:** " + ", ".join(detected_conditions))
    else:
        st.success("No significant findings detected above threshold.")

    # split layout into two secions
    col_chart, col_table = st.columns([3, 2])


    # create a horizontal bar chart, showcasing the probability of each diagnosis
    with col_chart:
        st.markdown("**Probability by condition**")

        # set bar colors (red if probability >= threshold, blue otherwise)
        colors = ["#d62728" if p >= threshold else "#1f77b4" for p in df["Probability"]]
        fig = plt.figure(figsize=(7, 6))
        ax = df["Probability"].plot(kind="barh", color=colors)
        ax.bar_label(ax.containers[0], labels=[f"{p*100:.1f}%" for p in df["Probability"]], padding=3) # show percentage at each bar
        plt.axvline(threshold, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
        plt.xlim(0, 1)
        plt.xlabel("Probability")
        plt.title(f"{model_name} Predictions")
        plt.legend(loc="lower right")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    # create a table higlighiting the probability and diagnosis of each condition
    with col_table:
        st.markdown("**All probabilities**")
        styled = df[["Condition", "Probability", "Detected"]].copy() # set column headers
        styled["Probability"] = styled["Probability"].map("{:.1%}".format)
        styled["Detected"] = styled["Detected"].map({True: "Yes", False: "No"})
        st.dataframe(styled, use_container_width=True, hide_index=True)


# create settings side bar, model selection drop down, and model descriptions
with st.sidebar:
    st.title("Settings")

    # model selection drop down
    model_name = st.selectbox(
        "Model",
        ["ResNet18", "DenseNet121", "CNN", "MLP"],
        help="Select the model to use for diagnosis.",
    )

    st.divider()

    # model descriptions
    st.markdown(
        """
**Models**
- **ResNet18 (Pretrained)**  - a CNN with 18 layers using residual connections to ease training deep networks, mainly for image classification; lightweight and fast.
- **DenseNet121 (Pretrained)** - a CNN with 121 layers where each layer connects to all previous layers, allowing for feature reuse, gradient flow, and efficiency
- **CNN (Custom)** - a custom 3-layer convolutional network trained on grayscale 128×128 chest X-rays from scratch.
- **MLP (Custom)** - a lightweight multi-layer perceptron trained on patient tabular metadata such as age, gender, view position, and follow-up number rather than images.
        """
    )


# page body
st.title("Chest X-Ray Diagnosis Predictor")
st.caption(
    "Upload a chest X-ray image. The selected model will predict the probability "
    "of 15 possible conditions from the [2017 NIH chest x-ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data?select=Data_Entry_2017.csv)."
)

# MLP Form
if model_name == "MLP":
    st.info("The MLP model uses patient metadata instead of an image. Fill in the fields below.")

    # Make form table using streamlit
    # 2 x 2 Table
    with st.form("mlp_form"):
        col1, col2 = st.columns(2)

        # Left column (Patient Age and Follow-up Number)
        with col1:
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=50) # set default patient age to 50
            follow_up = st.number_input("Follow-up Number", min_value=0, max_value=100, value=0)
        
        # Right column (Patient Gender and View Position)
        with col2:
            patient_gender = st.selectbox("Patient Gender", ["M", "F"])
            view_position  = st.selectbox("View Position",  ["PA", "AP"])

        # Submit buttion    
        submitted = st.form_submit_button("Run Diagnosis", type="primary", use_container_width=True)

    if submitted:
        model, err = load_model(model_name)

        # Display error message on error
        if err:
            st.error(err)
        else:
            # Encode inputs to match training preprocessing
            gender_enc = 1.0 if patient_gender == "M" else 0.0
            view_enc   = 1.0 if view_position  == "PA" else 0.0
            
            # Convert to Tensor
            features = torch.tensor(
                [[follow_up, float(patient_age), gender_enc, view_enc]],
                dtype=torch.float32,
            )

            with st.spinner("Running inference…"):
                probs = predict(model, features) # Predict using model and retrieve probabilities
            show_results(probs, model_name) # Display results

# Image dependent models: CNN, ResNet18, DenseNet121
else:
    # create file upload optionality
    uploaded = st.file_uploader(
        "Upload a chest X-ray (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible",
    )

    # check if file was uploaded
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB") # open file and convert to rgb

        col_img, col_info = st.columns([1, 2]) # create image and info column

        with col_img:
            st.image(pil_img, caption="Uploaded X-ray", use_container_width=True) # display image

        with col_info:
            st.markdown(f"**Selected model:** {model_name}") # show selected model
            st.markdown(f"**Image size:** {pil_img.size[0]} × {pil_img.size[1]} px")

        # add a run diagnosis confirmation button
        if st.button("Run Diagnosis", type="primary", use_container_width=True):
            model, err = load_model(model_name)

            # display error if model fails
            if err:
                st.error(err)
            else:
                with st.spinner("Running inference…"):
                    tensor = preprocess_image(pil_img, model_name)
                    probs  = predict(model, tensor) # run model to get probabilities
                show_results(probs, model_name) # display results
    else:
        st.info("Please upload a chest X-ray image.")
