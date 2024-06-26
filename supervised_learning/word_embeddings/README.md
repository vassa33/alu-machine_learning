# Natural Language Processing - Word Embeddings
![image](https://github.com/vassa33/alu-machine_learning/assets/61325877/5a144e23-1b17-4bd2-9ef7-5a2c364541ee)

This directory contains work with word embeddings as part of natural language processing (NLP):

## Mandatory Tasks:
0. [Bag of Words](/supervised_learning/0x0F-word_embeddings/0-bag_of_words.py)
* Write a function that creates a bag of words embedding matrix.
1. [TF-IDF](/supervised_learning/0x0F-word_embeddings/1-tf_idf.py)
* Write a function that creates a TF-IDF embedding.
2. [Train Word2Vec](/supervised_learning/0x0F-word_embeddings/2-word2vec.py)
* Write a function that creates and trains a gensim word2vec model.
3. [Extract Word2Vec](/supervised_learning/0x0F-word_embeddings/3-gensim_to_keras.py)
* Write a function that converts a gensim word2vec model to a keras Embedding layer.
4. [FastText](/supervised_learning/0x0F-word_embeddings/4-fasttext.py)
* Write a function that creates and trains a gensim fastText model.
5. [ELMo](/supervised_learning/0x0F-word_embeddings/5-elmo.py)
* Write a text file that answers multiple choice question about training an ELMo model.
When training an ELMo embedding model, you are training:

The internal weights of the BiLSTM
The character embedding layer
The weights applied to the hidden states
In the text file 5-elmo, write the letter answer, followed by a newline, that lists the correct statements:

A. 1, 2, 3
B. 1, 2
C. 2, 3
D. 1, 3
E. 1
F. 2
G. 3
H. None of the above
