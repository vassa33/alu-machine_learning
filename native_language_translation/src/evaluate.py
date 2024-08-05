from keras.models import load_model
import numpy as np

model = load_model("../models/rnn_model.h5")
test_data = np.load("../data/test_data.npz")


print("Model evaluation complete.")
