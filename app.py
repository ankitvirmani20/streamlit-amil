import streamlit as st
import pathlib
from PIL import Image
import google.generativeai as genai
import streamlit as st
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from datetime import datetime
from pathlib import Path
import os
import io
import subprocess

def install_requirements(file_path="requirements.txt"):
    """
    Installs Python packages listed in the specified requirements file.
    """

    try:
        result = subprocess.run(
            ["pip", "install", "-r", file_path],
            capture_output=True,
            text=True,
            check=True  # Raise an exception if pip fails
        )

        print("Installation successful!")
        print(result.stdout)  # Print pip's output for details

    except subprocess.CalledProcessError as e:
        print("Error during installation:")
        print(e.stderr)  # Show pip's error messages

    except FileNotFoundError:
        print(f"Error: Requirements file '{file_path}' not found.")

# Example usage:
install_requirements()  # Uses the default "requirements.txt"
install_requirements("requirements.txt")  #

# Configure the API key directly in the script
API_KEY = 'AIzaSyBpxyNMnfcnq2MZ6NhPj_T9sn_qwYSEHK8'
project_id = "powerful-star-422214-r6"
genai.configure(api_key=API_KEY)

# Set the environment variable immediately (make sure the path is correct)
project_id = "powerful-star-422214-r6"  # Replace with your project ID
vertexai.init(project=project_id, location="us-central1")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project_key.json"

# Generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Model name
MODEL_NAME = "gemini-1.5-pro-latest"

# Framework selection (e.g., Tailwind, Bootstrap, etc.)
framework = "Regular CSS use flex grid etc"  # Change this to "Bootstrap" or any other framework as needed

# Create the model
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat(history=[])


# Streamlit app
def main():
    image_path = "fukui.jpg"
    with open(image_path, "rb") as f:
        image_data = f.read()
    img = Image.open(io.BytesIO(image_data))
    st.image(img, caption="福井の永平寺", use_column_width=True)
    st.title("AIによる福井観光の生成分析")
    st.subheader('で作った ❤️ そして AI')

    survey_2024 = pd.read_csv('2024_happiness.csv')

    # User Input
    prompt = st.text_area(
        "Enter your analysis prompt:",
        "tell me some insights about how to increase tourism in Japanese",
    )

    model = GenerativeModel(model_name="gemini-1.5-pro")
    chat = model.start_chat()

    # --- Helper Functions ---
    def get_chat_response(chat: ChatSession, prompt: str) -> str:
        text_response = []
        responses = chat.send_message(prompt, stream=True)
        for chunk in responses:
            text_response.append(chunk.text)
        return "".join(text_response)

    # Load Data
    data_folder = Path("/workspaces/streamlit-fukui/") 
    try:

        # Prompt Generation (Adapted for Gemini and Streamlit Input)
        prompt = f"""Please answer the Question: {prompt} within 250 words. 
        Format your answer as bullet points like 1,2,3 etc, not paragraph and use this data:
        :\n{[survey_2024]}
        """  # Add the user's prompt here

        # Display Results
        st.subheader("Gemini-1.5-Pro Analysis:")
        with st.spinner("Analyzing..."):
            response = get_chat_response(chat, prompt)
            st.write(response)

    except FileNotFoundError:
        st.error(
            f"Data not found."
        )

if __name__ == "__main__":
    main()
