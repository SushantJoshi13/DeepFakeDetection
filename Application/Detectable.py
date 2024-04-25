import streamlit as st
import numpy as np
import pandas as pd
import pymongo
import torch
from torchvision import transforms, models
import cv2
from PIL import Image
import os
from collections import Counter
import urllib.parse


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():    
    # Retrieve MongoDB connection details
    username = urllib.parse.quote_plus(st.secrets["mongo"]["username"])
    password = urllib.parse.quote_plus(st.secrets["mongo"]["password"])
    cluster = urllib.parse.quote_plus(st.secrets["mongo"]["cluster"])
    return pymongo.MongoClient(f"mongodb+srv://%s:%s@%s.kzh1moi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" % (username, password, cluster))

client = init_connection()
db = client.users

# Define the model
class DeepfakeModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeepfakeModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@st.cache_data()
def load_model(path_to_model):
    model = DeepfakeModel(2)
    state_dict = torch.load(path_to_model, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

# Define transforms
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict_deepfake(model, video_path):
    vidObj = cv2.VideoCapture(video_path)
    success = 1
    predictions = []
    frame_count = 0
    prev_prediction = None

    while success:
        success, image = vidObj.read()
        if success:
            # Convert OpenCV image to PIL format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)

            # Apply the same transformations as during training
            image_tensor = transform(image_pil).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                logits = model(image_tensor)
                _, prediction = torch.max(logits, 1)

            predictions.append(prediction.item())

            # Save frame if prediction changes
            if prev_prediction is not None and prev_prediction != prediction.item():
                output_path = f"frame_{frame_count}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            prev_prediction = prediction.item()
            frame_count += 1

    # Determine the majority vote
    counter = Counter(predictions)
    majority_vote = counter.most_common(1)[0][0]

    return "REAL" if majority_vote == 1 else "FAKE"

def home_page():
    # Streamlit app title and layout
    st.title("Detectable")
    st.markdown("---")

    st.write("Welcome to Detectable! Upload an image or video to verify its authenticity.")
    # File uploader
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    
    # Check if user is logged in using local storage
    is_logged_in = st.session_state.get("logged_in", False)

    # Verification button
    if is_logged_in:
        if st.button("Verify"):
            if uploaded_file is not None:
                # Save uploaded video
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display uploaded video
                st.video("temp_video.mp4")

                # Load the model
                model = load_model('model_97_acc_80_frames_FF_data.pt')
                
                # Make prediction and show the spinner until the prediction is ready
                with st.spinner("Detecting deepfake..."):
                    prediction = predict_deepfake(model, "temp_video.mp4")

                # Display final prediction with emoji
                emoji = "✅" if prediction == "REAL" else "❌"
                st.write(f"Video is {prediction} {emoji}")

                # Remove temporary video file
                os.remove("temp_video.mp4")
                # delete all the generated images from the directory
                for file in os.listdir():
                    if file.startswith("frame"):
                        os.remove(file)                  
            else:
                st.error("Please upload a image or video first.")
    else:
        st.write("Please login to enable image upload.")
    

def about_page():
    st.title("About Our Deepfake Detection App")

    # Optional: Display information about the app using a DataFrame (replace with your data)
    about_data = pd.DataFrame({
        "Feature": ["Deep Learning Model", "Accuracy", "Functionality"],
        "Value": ["Convolutional Neural Network (CNN)", "90% (placeholder)", "Detects deepfakes in images and videos"]
    })
    st.table(about_data)

    # Add more details about the app, creators, etc.
    st.write("This app is designed to help users identify potential deepfakes...")

def login_page():
    st.title("Login")

    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login")
    
    if login_button:
        # Check for empty fields
        if not username or not password:
            st.error("Please enter a username and password")
            return
        
        user = db.mycollection.find_one({"username": username, "password": password})
        # if authentication is successful
        if user:
            # Set the session state to logged in
            st.session_state.logged_in = True
            # Add page_name Logout to the sidebar and remove Login
            page_names.remove("Login")
            page_names.remove("Register")
            page_names.append("Logout")
            # Display a success message
            st.success("Login successful. Welcome, " + username + "!")
            # Redirect to home page
            page_selection = "Home"
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Please try again.")
            

# Logout function
def logout():
    # Clear the session state
    st.session_state.logged_in = False   
    # Remove page_name Logout from the sidebar and add Login
    page_names.remove("Logout")
    page_names.append("Login")

    # Display a message if the user is logged out successfully
    if not st.session_state.get("logged_in", False):
        st.write("You have been logged out successfully.")
        # Redirect to home page
        page_selection = "Home"
        st.experimental_rerun()
    else:
        st.write("An error occurred. Please try again.")

# Register the user
def register():
    st.title("Register")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    confirm_password = st.text_input("Confirm Password:", type="password")
    register_button = st.button("Register")

    if register_button:
        # Check for empty fields
        if not username or not password or not confirm_password:
            st.error("Please fill in all fields")
            return
        # Check if passwords match
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        # Check if user already exists
        user = db.mycollection.find_one({"username": username})
        if user:
            st.error("User already exists. Please login.")
            return
        # If user does not exist, add user to the database
        if not user:
            result = db.mycollection.insert_one({"username": username, "password": password})
            # check if user is added successfully
            if result is not None:
                # If user is added successfully, display success message
                st.success("User registered successfully. Please login to continue.")
                # Redirect to login page
                page_selection = "Login"
                st.experimental_rerun()
            else:
                st.error("An error occurred. Please try again.")
        else:
            st.error("An error occurred. Please try again.")

# Create the multipage app structure
st.sidebar.title("Navigation")
page_names = ["Home", "About"]
 # Check if user is logged in using local storage
is_logged_in = st.session_state.get("logged_in", False)
if is_logged_in:
    page_names.append("Logout")
else:
    page_names.append("Login")
    page_names.append("Register")
page_selection = st.sidebar.selectbox("Select a page", page_names)

if page_selection == "Home":
    home_page()
elif page_selection == "About":
    about_page()
elif page_selection == "Register":
    register()
elif page_selection == "Login":
    if st.session_state.get("logged_in", False):
        st.write("You are already logged in.")
    else:
        login_page()
elif page_selection == "Logout":
    logout()



