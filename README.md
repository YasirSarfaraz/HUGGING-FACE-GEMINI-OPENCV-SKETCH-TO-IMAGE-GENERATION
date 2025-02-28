# UNDER DEVELOPMENT

The project is still in progress there are many things to Add and modify.
This is not the final solution.

# HUGGING-FACE-GEMINI-OPENCV-SKETCH-TO-IMAGE-GENERATION

HUGGING-FACE-GEMINI-OPENCV-SKETCH-TO-IMAGE-GENERATION is an AI-powered interactive drawing tool that allows users to create digital sketches using hand gestures. It utilizes computer vision for gesture detection and integrates generative AI to convert sketches into detailed images. The project is built using Streamlit, OpenCV, MediaPipe, and Hugging Face’s Stable Diffusion API.

## Features

- **Hand Gesture Drawing**: Uses MediaPipe Hands to detect finger movements and enable intuitive drawing.
- **Gesture-Based Controls**: Supports different gestures for drawing, erasing, and clearing the canvas.
- **AI-Powered Sketch Interpretation**:  Converts user drawings into text descriptions using Google Gemini AI.
- **Image Generation**: Generates high-quality images from sketches using Stable Diffusion XL from Hugging Face.
- **Streamlit UI**: A simple, interactive web application for users to draw and generate images.

## Approach Adopted
### Hand Tracking & Gesture Recognition:
   - Used MediaPipe Hands for real-time finger tracking.
   - Mapped different hand gestures to specific drawing actions.
### Sketch Interpretation:
   - Converted sketches into meaningful text descriptions using Google Gemini AI.
   - Used image preprocessing techniques to enhance input clarity.
### Image Generation:
   - Utilized Hugging Face’s Stable Diffusion XL to generate high-quality images based on interpreted prompts.
   - Optimized model inputs to improve coherence between the sketch and generated image.

## Technology Stack

- **Python 3.10.11**     
- **Streamlit (for UI)**
- **OpenCV (for image processing)**
- **MediaPipe Hands (for gesture tracking)**
- **Hugging Face API (for AI image generation)**
- **Google Gemini AI (for sketch understanding)**
- **NumPy, PIL (for image manipulation)**

## Models Used
- Google Gemini AI: Used for interpreting hand-drawn sketches and converting them into descriptive text prompts.
- Hugging Face Stable Diffusion XL: Generates high-resolution images based on the processed text prompts.
- MediaPipe Hands: Detects and tracks hand gestures to enable interactive drawing.

## Architecture
![Architecture_HuggingFace](https://github.com/user-attachments/assets/1d7b343a-1c32-4c75-8e0d-36dedd619f67)


## Installation & Setup
### 1. Clone the Repository
- git clone https://github.com/yourusername/HUGGING-FACE-GEMINI-OPENCV-SKETCH-TO-IMAGE-GENERATION.git cd HUGGING-FACE-GEMINI-OPENCV-SKETCH-TO-IMAGE-GENERATION
### 2. Install Dependencies
- pip install -r requirements.txt
### 3. Set Up Environment Variables
- Create a .env file in the root directory and add:
- HUGGING_FACE_API_KEY=your_api_key
- GOOGLE_API_KEY=your_google_api_key
### 4. Run the Application
- streamlit run genai.py

## Usage
1. Start the application and allow webcam access.
2. Use hand gestures to draw on the screen.
   - Index & Middle Finger Up: Draw
   - Index, Middle & Ring Finger Up: Lift pen (stop drawing)
   - Index & Pinky Up: Clear the screen
   - Thumb & Index Up: Erase
3. Generate an AI image by clicking on "Generate Image."
4. Download the final generated image.

## Project Showcase
### 1. Live Demo (Video)


### 2. Sample Outputs (Screenshots)



## Future Improvements
- Improve sketch interpretation with better AI models.
- Enhance gesture recognition for more natural controls.
- Support multiple drawing tools (brush size, colors, etc.).
- Optimize processing speed for real-time feedback.
