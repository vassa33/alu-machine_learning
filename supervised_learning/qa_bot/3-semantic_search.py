#!/usr/bin/env python3
"""
Find a snippet of text within a reference document to answer a question
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Function to searches multiple articles
    corpus_path: path to the docs
    sentence: question or sentence trying to find best match
    """
    m = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    articles = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            articles.append(f.read())

    embeddings = m(articles)

    corr = np.inner(embeddings, embeddings)

    closest = np.argmax(corr[0, 1:])

    return articles[closest + 1]
