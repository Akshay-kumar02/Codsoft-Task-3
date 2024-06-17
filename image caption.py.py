# Install necessary libraries if you haven't already
# !pip install tensorflow numpy matplotlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

# Load pre-trained InceptionV3 model (without classification layers)
image_model = InceptionV3(include_top=False, weights='imagenet')

# Remove the last convolutional layer
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = Model(inputs=new_input, outputs=hidden_layer)

# Load and preprocess an image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract image features
def extract_image_features(image_path):
    img = load_and_preprocess_image(image_path)
    image_features = image_features_extract_model.predict(img)
    image_features = tf.reshape(image_features, (image_features.shape[0], -1, image_features.shape[3]))
    return image_features

# Define the captioning model
def create_model(vocab_size, max_length):
    input_image = Input(shape=(None, 2048))
    image_model = LSTM(256)(input_image)

    input_caption = Input(shape=(max_length,))
    caption_model = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
    caption_model = LSTM(256)(caption_model)

    decoder = tf.keras.layers.concatenate([image_model, caption_model])
    decoder = Dense(256, activation='relu')(decoder)
    decoder_output = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[input_image, input_caption], outputs=decoder_output)
    return model

# Load and preprocess captions
captions = ...  # Load your captions data here

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(s) for s in sequences)

# Pad sequences to uniform length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare training data
image_features = ...  # Load your image features data here
X1, X2, y = [], [], []
for i in range(len(padded_sequences)):
    sequence = padded_sequences[i]
    for j in range(1, len(sequence)):
        in_seq, out_seq = sequence[:j], sequence[j]
        in_seq = to_categorical([in_seq], num_classes=vocab_size)[0]
        X1.append(image_features[i])
        X2.append(in_seq)
        y.append(out_seq)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)

# Create the model
model = create_model(vocab_size, max_length)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([X1, X2], y, epochs=10, batch_size=64)

# Generate captions for new images
def generate_caption(image_path):
    image_features = extract_image_features(image_path)
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        sequence = to_categorical(sequence, num_classes=vocab_size)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Example usage
image_path = 'example.jpg'
caption = generate_caption(image_path)
print("Generated Caption:", caption)
