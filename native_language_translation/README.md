# English to Kiswahili Translation with RNNs

## Overview
This project develops a neural machine translation model using a Recurrent Neural Network (RNN) to translate English text to Kiswahili. The project utilizes an encoder-decoder architecture and is implemented in Python using TensorFlow and Keras.

## Project Structure
```
├── data/
│   ├── train.txt
│   ├── test.txt
│   └── processed_data.npz
├── models/
│   └── rnn_model.h5
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── eng2swa.ipynb
└── README.md
```

## Requirements
- Python
- TensorFlow
- Keras
- NumPy

To install the necessary libraries, run the following command:
```
pip install -r requirements.txt
```

## Setup Instructions

### Virtual Environment Setup
#### Linux:
```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
#### Windows:
```bash
pip install virtualenv
virtualenv venv
venv\Scripts\activate
```

### Installation of Dependencies
Ensure the virtual environment is active:
```
pip install -r requirements.txt
```

## Data Preparation
Data should be placed in the `data/` directory. `train.txt` and `test.txt` should contain English and Kiswahili sentence pairs separated by tabs. Run `src/preprocess.py` to preprocess the data:
```
python src/preprocess.py
```

## Model
The model is defined in `src/model.py`. It uses an LSTM-based encoder-decoder architecture with embedding layers. The encoder processes English sentences, and the decoder generates the corresponding Kiswahili translation.

## Training
To train the model, run:
```
python src/train.py
```
Training details and configurations can be adjusted in `src/train.py`.

## Evaluation
The model is evaluated using the BLEU score to assess the quality of the translations. Run:
```
python src/evaluate.py
```

## Jupyter Notebook
The notebook `eng2swa.ipynb` provides an interactive environment to run all the steps from setting up the environment, processing data, training the model, and evaluating it.
