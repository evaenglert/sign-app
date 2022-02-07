import streamlit as st
from PIL import Image

Py = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://www.python.org/"><h4 style='text-align: centre; color: black;font-size: 20px;'>Python</h4</a>
</div>
"""

TFC = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://www.tensorflow.org/api_docs/python/tf/keras/applications"><h4 style='text-align: centre; color: black;font-size: 20px;'>Tensorflow Classification</h4</a>
</div>
"""

TFD = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md"><h4 style='text-align: centre; color: black;font-size: 20px;'>Tensorflow Object Detection</h4</a>
</div>
"""

sl = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://www.streamlit.io/"><h4 style='text-align: centre; color: black;font-size: 20px;'>Streamlit</h4</a>
</div>
"""

hr = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://www.heroku.com/"><h4 style='text-align: centre; color: black;font-size: 20px;'>Heroku</h4</a>
</div>
"""

gcp = """
<div style="text-align: centre" class="contributors font-body text-bold">
<a class="contributor comma" href="https://console.cloud.google.com/"><h4 style='text-align: centre; color: black;font-size: 20px;'>Google cloud engine</h4</a>
</div>
"""

# pylint: disable=line-too-long
def write():
    st.title("Technology stack")
    st.write("##")
    st.write("##")
    st.write("##")
    col1, col2, col3, col4,col5,col6 = st.columns([1,1,1,1,1,1])

    with col1:
        image = Image.open('images/python.jpeg')
        st.image(image, caption='python', width=80, use_column_width=False)
        # st.markdown(Py, unsafe_allow_html=True)
    with col2:
        image = Image.open('images/tf.png')
        st.image(image, caption='Tensorflow classification', width=300, use_column_width=True)
        # st.markdown(TFC, unsafe_allow_html=True)

    with col3:
        image = Image.open('images/tf.png')
        st.image(image, caption='Tensorflow object detection', width=300, use_column_width=True)
        # st.markdown(TFD, unsafe_allow_html=True)

    with col4:
        image = Image.open('images/streamlit.png')
        st.image(image, caption='Streamlit', width=100, use_column_width=False)
        # st.markdown(sl, unsafe_allow_html=True)
    with col5:
        image = Image.open('images/heroku.png')
        st.image(image, caption='Heroku', width=200, use_column_width=False)
        # st.markdown(hr, unsafe_allow_html=True)

    with col6:
        image = Image.open('images/gcp.png')
        st.image(image, caption='Google cloud engine', width=200, use_column_width=False)
