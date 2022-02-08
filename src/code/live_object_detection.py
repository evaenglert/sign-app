import streamlit as st
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import queue

#Import for streamlit
import streamlit as st
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#Import for Deep learning model
import tensorflow as tf

#Import for handling image
import cv2
from cvzone.HandTrackingModule import HandDetector


#Create a dict for classes <- got rid of this for now
# @st.cache(allow_output_mutation=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{
        "urls": ["stun:stun.l.google.com:19302"]
    }]})

# CWD path
HERE = Path(__file__).parent


#Main intelligence of the file, class to launch a webcam, detect hands, then detect sign and output american letters
def app_object_detection(model=[],label=[]):

    class SignPredictor(VideoProcessorBase):


        def __init__(self) -> None:
            # Hand detector
            self.hand_detector = HandDetector(detectionCon=0.5, maxHands=1)

            # Sign detector
            self.model = model
            self.label = label

            #Queue to share information that happens within the live video thread outside the thread
            self.result_queue = queue.Queue()

        def find_hands(self, image):

            hands = self.hand_detector.findHands(image, draw=False)

            # loop over all hands and print them on the video + apply predictor
            for hand in hands:
                # this is just an array of len 4, containing info about the bounding box
                bbox = hand["bbox"]
                x, y, w, h = bbox

                # .rectangle needs the image, the top right, bottom left points of the rectangle,
                # and color of the rectangle and line thickness
                if bbox[2] > bbox[3]:
                    h = w
                    diff = int((bbox[2] - bbox[3]) / 2)

                    rectangle = cv2.rectangle(
                        image, (bbox[0] - 20, bbox[1] - 20 - diff),
                        (bbox[0] + bbox[2] + 20,
                        bbox[1] + bbox[3] + 20 + diff), (0, 0, 0), 2)

                    cropped_image = image[max(0, y - 20 - diff):y + h + 20 +
                                        diff,
                                        max(0, x - 20):x + w + 20]

                else:
                    diff = int((bbox[3] - bbox[2]) / 2)
                    rectangle = cv2.rectangle(
                        image, (bbox[0] - 20 - diff, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20 + diff,
                        bbox[1] + bbox[3] + 20), (0, 0, 0), 2)

                    cropped_image = image[max(0, y - 20):y + h + 20,
                                        max(0, x - 20 - diff):x + w + 20 +
                                        diff]


                prediction = self.model.predict(
                    np.array(
                        tf.image.resize(
                            (cropped_image), [128, 128]) / 255).reshape(
                                -1, 128, 128, 3))

                prediction_max = np.argmax(prediction)

                pred = self.label[prediction_max]
                #check on terminal the prediction
                print(pred)
                #store prediction on the queue to use them outside of the live thread
                self.result_queue.put(pred)

                #draw letter on images
                cv2.putText(rectangle, pred, (bbox[0] + 30, bbox[1] - 30),
                            cv2.FONT_ITALIC, 2, (0, 0, 0), 2)


            return hands


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            self.find_hands(image)
            return av.VideoFrame.from_ndarray(cv2.flip(image, 1), format="rgb24")



    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=SignPredictor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    # if st.checkbox("Show the detected labels", value=True):
    #     if webrtc_ctx.state.playing:
    #         labels_placeholder = st.empty()
    #         # NOTE: The video transformation with object detection and
    #         # this loop displaying the result labels are running
    #         # in different threads asynchronously.
    #         # Then the rendered video frames and the labels displayed here
    #         # are not strictly synchronized.
    #         while True:
    #             if webrtc_ctx.video_processor:
    #                 try:
    #                     result = webrtc_ctx.video_processor.result_queue.get(
    #                         timeout=1.0)
    #                 except queue.Empty:
    #                     result = None
    #                 labels_placeholder.write(result)
    #             else:
    #                 break
