import streamlit as st
import src.code.live_object_detection as lod
# pylint: disable=line-too-long
#Import for Deep learning model
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
import subprocess

if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/evaenglert/sign-app/master/saved_models/asl_model.h5"'], shell=True)



#deep learning sign detector model cached
@st.cache(allow_output_mutation=True)
def retrieve_model(PATH_MODEL, PATH_LABEL):
    """ dummy tensorflow CNN model trained on few epochs on multiclassification task (american signs) """

    with open(PATH_LABEL, "rb") as fp:
        label = pickle.load(fp)
#     model = load_model(PATH_MODEL)
    model = load_model('model.h5', compile=False)
    # label = np.load(PATH_LABEL, allow_pickle=True)
    return model, label


# # print(os.getcwd())
PATH_MODEL_ASL = "saved_models/asl_model.h5"
PATH_LABEL_ASL = "saved_labels/asl_class_names.txt"

model_asl, label_asl = retrieve_model(PATH_MODEL_ASL, PATH_LABEL_ASL)


def write(mas=model_asl,
              lasl=label_asl):

    options = st.selectbox("Choose what you want to test today",
                           index=0,
                           options=[
                               "Live demo with American sign"
                           ])

    if options=="Live demo with American sign":
        st.write("##")
        st.markdown(
            "<h4 style='text-align: center; color: black;'>Click on the start button to activate the webcam</h4",
                    unsafe_allow_html=True)
        st.write("##")
        lod.app_object_detection(model=mas, label=lasl)


write(mas=model_asl,
      lasl=label_asl)
