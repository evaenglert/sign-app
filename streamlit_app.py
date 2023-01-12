import streamlit as st

st.set_page_config(
    page_title="Sign Language App",  # => Quick reference - Streamlit
    layout="wide",  # wide
    initial_sidebar_state="auto")  # collapsed


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import queue
import random
#Import for streamlit
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from PIL import Image

#Import for Deep learning model
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import src.pages.resources
import src.pages.tryme

if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/evaenglert/sign-app/blob/master/saved_models/asl_model.h5"'], shell=True)


PAGES = {
    "Video demo": src.pages.tryme,
    "Tech stack": src.pages.resources,

}

# """Main function of the App"""
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]

with st.spinner(f"Loading {selection} ..."):
    page.write()


#Import for handling image
import cv2
from cvzone.HandTrackingModule import HandDetector


############################ Sidebar + launching #################################################

st.sidebar.title("Contribute")
st.sidebar.info(
    "Please find the source code for this project [here](https://github.com/evaenglert/flykr). ")
