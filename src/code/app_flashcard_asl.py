from PIL import Image
import numpy as np
import streamlit as st
import requests
import json
import random
from typing import TypeVar
import dataclasses
import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow
import tensorflow as tf
from tensorflow.python import tf2
import pickle

HI = 1000

StateT = TypeVar('StateT')


def persistent_game_state(initial_state: StateT) -> StateT:
    session_id = st.report_thread.get_report_ctx().session_id
    session = st.server.server.Server.get_current()._get_session_info(
        session_id).session
    if not hasattr(session, '_gamestate'):
        setattr(session, '_gamestate', initial_state)
    return session._gamestate


# Function to Read and Convert Images
def load_image(img):
    im = Image.open(img)
    im = im.resize((256, 256), Image.ANTIALIAS)
    image = np.array(im)
    return image


with open("saved_labels/asl_class_names2.txt", "rb") as fp:
    all_signs = pickle.load(fp)

letter_to_guess = all_signs[random.randint(0, len(all_signs)-1)]

@dataclasses.dataclass
class GameState:
    number: int
    num_guesses: int = 0
    game_number: int = 0
    game_over: bool = False

# callback to update emojis in Session State
# in response to the on_click event
def random_emoji():
    st.session_state.emoji = random.choice(emojis)

# initialize emoji as a Session State variable
if "emoji" not in st.session_state:
    st.session_state.emoji = "👈"

emojis = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻"]


def flashcard(model=[],label=[]):
    GS = GameState(random.randint(1, len(all_signs) - 1))
    state = persistent_game_state(initial_state=GS)


    if st.button("NEW LETTER"):
        state.number = random.randint(1, len(all_signs) - 1)
        state.num_guesses = 0
        state.game_number += 1
        state.game_over = False

    if not state.game_over:

        letter_to_guess = all_signs[state.number]
        st.write("Do you know the sign for " + letter_to_guess.upper() + "?")
        #uploadFile = st.file_uploader(label="Please take a picture :)", type=['jpg', 'png'])

        if st.button('Take picture:)'):

            cap = cv2.VideoCapture(0)
            detector = HandDetector(detectionCon=0.8, maxHands=2)
            counter = 0
            hand = None
            while hand is None:
                # Get image frame
                success, img = cap.read()
                # if counter % 5 == 0:
                raw_image = img.copy()
                # Find the hand and its landmarks
                hands, img = detector.findHands(img)  # with draw
                # hands = detector.findHands(img, draw=False)  # without draw
                st.image(raw_image)


            # Display
            # counter += 1
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
            cap.release()
            cv2.destroyAllWindows()
            # camera = cv2.VideoCapture(0)
            # hand_detector = HandDetector(detectionCon=0.5, maxHands=1)

            # # success, img = cap.read()
            # def get_image():
            #     hand = None
            #     while hand is None:
            #         retval, im = camera.read()
            #         hand = hand_detector.findHands(im, draw=False)
            #     return im
            # for i in range(30):
            #     temp = camera.read()

            # img = get_image()

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # st.image(img)
            # print(img.shape)
            # camera.release()
            # cv2.destroyAllWindows()

            # hand = hand_detector.findHands(img, draw=False)
            # bbox = hand[0]["bbox"]
            # x, y, w, h = bbox
            # # image_to_classify = img[y:y+h, x:x+w]


            # if bbox[2] > bbox[3]:
            #     diff = int((bbox[2] - bbox[3]) / 2)

            #     rectangle = cv2.rectangle(
            #         img, (bbox[0] - 20, bbox[1] - 20 - diff),
            #         (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20 + diff),
            #         (0, 0, 0), 2)

            #     cropped_image = img[max(0, y - 20 - diff):y + h + 20 + diff,
            #                         max(0, x - 20):x + w + 20]

            # else:
            #     diff = int((bbox[3] - bbox[2]) / 2)
            #     rectangle = cv2.rectangle(
            #         img, (bbox[0] - 20 - diff, bbox[1] - 20),
            #         (bbox[0] + bbox[2] + 20 + diff, bbox[1] + bbox[3] + 20),
            #         (0, 0, 0), 2)

            #     cropped_image = img[max(0, y - 20):y + h + 20,
            #                         max(0, x - 20 - diff):x + w + 20 + diff]

            # imgage_resized = np.array(
            #     tf.image.resize((cropped_image), [128, 128]) / 255)

            # prediction = model.predict(
            #         np.array(
            #             tf.image.resize(
            #                 (cropped_image), [128, 128]) / 255).reshape(
            #                     -1, 128, 128, 3))
            # st.image(cropped_image)
            # prediction_max = np.argmax(prediction)
            # pred = label[prediction_max]

            # if pred == letter_to_guess.lower():
            #     st.write(
            #         f"Well, that was a super duper guess, this is indeed the right sign for {letter_to_guess.upper()} :) So smart. 🎉🎉🎉"
            #     )
            # else:
            #     st.write(
            #         f"Good try, but actually, this is more like a {pred}. But practice makes perfect 😏!"
            #     )
