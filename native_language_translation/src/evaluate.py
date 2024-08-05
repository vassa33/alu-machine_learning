from keras.models import load_model
import numpy as np

model = load_model("../models/rnn_model.h5")
test_data = np.load("../data/test_data.npz")  # Assume this is prepared

# Add evaluation code here, perhaps calculating BLEU score

print("Model evaluation complete.")
