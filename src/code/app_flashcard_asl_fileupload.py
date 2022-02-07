from PIL import Image
import numpy as np
import streamlit as st
import random
from typing import TypeVar
import dataclasses

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
    im = im.resize((128, 128), Image.ANTIALIAS)
    image = np.array(im)
    return image


all_signs = ["a", "b", "o", "5", "0", "c"]
#all_signs = np.load("saved_models/asl_class_names2.npy")

letter_to_guess = all_signs[random.randint(0, len(all_signs)-1)]

@dataclasses.dataclass
class GameState:
    number: int
    num_guesses: int = 0
    game_number: int = 0
    game_over: bool = False


def flashcard(model=[], label=[]):
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
        uploadFile = st.file_uploader(label="Upload your guess :) ", type=['jpg', 'png'])

        if uploadFile  is not None:
            # Perform  Manipulations
            img = load_image(uploadFile)
            st.image(img)
            st.write(":camera_with_flash: Image Uploaded Successfully !")
            # Reshape the image

            prediction = model.predict(
                    np.array((img[:, :, :3]) / 255).reshape(
                                -1, 128, 128, 3))
            prediction_max = np.argmax(prediction)
            pred = label[prediction_max]


            # e.g bruschetta
            if pred == letter_to_guess.lower():
                st.write(
                    f"Well, that was a super duper guess, this is indeed the right sign for {letter_to_guess.upper()} :) So smart. 🎉🎉🎉"
                )
            else:
                st.write(
                    f"Good try, but actually, this is more like a {pred}. But practice makes perfect 😏!"
                )

        else:
            st.write("Make sure you image is in JPG/PNG Format.")
