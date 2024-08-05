import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(filepath):
    with open(filepath, encoding='utf-8') as file:
        lines = file.read().split('\n')
    pairs = [line.split('\t') for line in lines if line != ""]
    eng_texts = [pair[0] for pair in pairs]
    swa_texts = ['\t' + pair[1] + '\n' for pair in pairs]  # Start and end tokens
    return eng_texts, swa_texts

def tokenize(texts):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def preprocess_data(eng_texts, swa_texts):
    eng_tokenizer = tokenize(eng_texts)
    swa_tokenizer = tokenize(swa_texts)
    eng_sequences = eng_tokenizer.texts_to_sequences(eng_texts)
    swa_sequences = swa_tokenizer.texts_to_sequences(swa_texts)
    eng_data = pad_sequences(eng_sequences, padding='post')
    swa_data = pad_sequences(swa_sequences, padding='post')
    return eng_data, swa_data, eng_tokenizer, swa_tokenizer

if __name__ == "__main__":
    train_eng_texts, train_swa_texts = load_data("../data/train.txt")
    test_eng_texts, test_swa_texts = load_data("../data/test.txt")
    eng_data, swa_data, eng_tokenizer, swa_tokenizer = preprocess_data(train_eng_texts, train_swa_texts)
    np.savez_compressed("../data/processed_data.npz", eng_data=eng_data, swa_data=swa_data)
    print("Data preprocessed and saved.")
