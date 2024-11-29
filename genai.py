import streamlit as st
st.set_page_config(page_title='Image Generator', layout="wide")
import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import requests
import base64
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from io import BytesIO
from time import sleep
filterwarnings(action='ignore')
class ImageGenerator:
    def __init__(self):
        load_dotenv()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=950)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=550)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=130)

        self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)

        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        self.p1, self.p2 = 0, 0

        self.p_time = 0

        self.fingers = []

        self.api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        self.api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.headers = {"Authorization": "Bearer " + self.api_key}

        if not self.api_key:
            st.error("API key not found. Please set HUGGING_FACE_API_KEY in your environment.")

    def streamlit_config(self):
        page_background_color = """
        <style>
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        .block-container {
            padding-top: 0rem;
        }
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)

    def main(self):

        self.streamlit_config()

        st.title("Arithmo Draw")
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            
            stframe = st.empty()

        with col3:
            
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=300,  
            width=600,   
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            
            st.image(canvas_result.image_data, caption="Your Drawing")

            prompt = self.understand_drawing(canvas_result.image_data)
            st.write(f"Generated Prompt: {prompt}")

            generated_image = self.generate_image_from_prompt(prompt+"coloured image")
            if generated_image:
                result_placeholder.image(generated_image, caption="Generated Image")
            else:
                st.error("Image generation failed. Please try again.")

        while True:
            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown(
                    '<h4 style="text-align:center;color:orange;">Error: Could not open webcam. '
                    'Please ensure your webcam is connected and try again.</h4>',
                    unsafe_allow_html=True,
                )
                break

            self.process_frame()
            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(self.img, channels="RGB")

            if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
       
                prompt = self.understand_drawing(self.imgCanvas)
                st.write(f"Generated Prompt: c {prompt}")

                generated_image = self.generate_image_from_prompt(prompt)
                if generated_image:
                    result_placeholder.image(generated_image, caption="Generated Image")
                else:
                    st.error("Image generation failed. Please try again.")

        self.cap.release()
        cv2.destroyAllWindows()


    def process_frame(self):
        success, img = self.cap.read()
        img = cv2.resize(src=img, dsize=(950,550))
        self.img = cv2.flip(src=img, flipCode=1)
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        result = self.mphands.process(image=self.imgRGB)

        self.landmark_list = []

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(image=self.img, landmark_list=hand_lms, 
                                            connections=hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = self.img.shape
                    x, y = lm.x, lm.y
                    cx, cy = int(x * w), int(y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        
        self.fingers = []

        if self.landmark_list != []:
            for id in [4,8,12,16,20]:

                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
       
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)

            for i in range(0, 5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1], self.landmark_list[(i+1)*4][2]
                    cv2.circle(img=self.img, center=(cx,cy), radius=5, color=(255,0,255), thickness=1)

    def handle_drawing_mode(self):
       
        if sum(self.fingers) == 2 and self.fingers[0]==self.fingers[1]==1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(255,0,255), thickness=5)

            self.p1,self.p2 = cx,cy
        
        elif sum(self.fingers) == 3 and self.fingers[0]==self.fingers[1]==self.fingers[2]==1:
            self.p1, self.p2 = 0, 0
        
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[2]==1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
        
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy

            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(0,0,0), thickness=15)

            self.p1,self.p2 = cx,cy
        
        elif sum(self.fingers) == 2 and self.fingers[0]==self.fingers[4]==1:
            self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)

        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)

        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)

        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        img = cv2.bitwise_and(src1=img, src2=imgInv)

       
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)

    def understand_drawing(self, image_data):
      
        imgCanvas = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        imgCanvas = PIL.Image.fromarray(imgCanvas)

        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

        prompt = "Analyze the drawing and provide a description."

        response = model.generate_content([prompt, imgCanvas])

        description = response.text if response.text else "A drawing"
        return description

    def generate_image_from_prompt(self, prompt):
        if not self.api_key:
            return None

        payload = {
            "inputs": prompt+"in color",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)  
                response.raise_for_status()  

                print("API Response:", response.content)

                image = Image.open(BytesIO(response.content))
                return image

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    sleep(5) 
                    continue
                st.error(f"An error occurred: {e}")
                return None

if __name__ == "__main__":
    try:
        generator = ImageGenerator() 
        generator.main()             
    except Exception as e:
        add_vertical_space(5)
        st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)