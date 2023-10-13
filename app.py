import streamlit as st
import pickle
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
from keras.preprocessing.text import Tokenizer
import numpy as np

# Load your tokenizer and model here
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

vocab_size = len(tokenizer.word_index)+1
max_length = 34

model = load_model('your_model.h5')



def extract_img_features(image):

    # Load the model
    vgg_model = VGG16()

    # Restructure the model
    model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

    # Resize the image to the desired target size
    image = image.resize((224, 224))

    # Convert image pixels to a numpy array
    image = img_to_array(image)

    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Prepare the image for the VGG model
    image = preprocess_input(image)

    # Get the features through the VGG model for our input
    feature = model.predict(image, verbose=0)

    return feature

def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):

    # Add the start tag for the generation process
    in_text = 'startseq'

    # Iterate over the max length of the sequence
    for i in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad the encoded sequence
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict the next word
        y_hat = model.predict([image, sequence], verbose=0)

        # Get the index with high probability
        y_hat = np.argmax(y_hat)

        # Convert the index to a word
        word = index_to_word(y_hat, tokenizer)

        # Stop if no word is found
        if word is None:
            break

        # Append the word as input for generating the next word
        in_text += " " + word

        # Stop if we reach endseq
        if word == 'endseq':
            break

    return in_text.replace("startseq", "").replace("endseq", "")

st.title("Image Captioning App")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Generate Caption"):
        feature = extract_img_features(image)
        caption = predict_caption(model, feature, tokenizer, max_length)
        st.write("Generated Caption:", caption)

