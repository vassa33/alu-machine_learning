import numpy as np
from model import create_model
from keras.callbacks import ModelCheckpoint

data = np.load("../data/processed_data.npz")
eng_data = data['eng_data']
swa_data = data['swa_data']

# Assume tokenizers saved and loaded here
num_encoder_tokens = 100  # Placeholder, should be replaced with actual value
num_decoder_tokens = 100  # Placeholder, should be replaced with actual value

model = create_model(num_encoder_tokens, num_decoder_tokens)
model.summary()

checkpoint = ModelCheckpoint("../models/rnn_model.h5", save_best_only=True)
model.fit([eng_data, swa_data[:, :-1]], np.expand_dims(swa_data[:, 1:], -1), batch_size=64, epochs=50, validation_split=0.2, callbacks=[checkpoint])
