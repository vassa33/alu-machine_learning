import json
import re
import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import streamlit as st

# Load the dataset
with open('dataset.json', 'r') as file:
    data = json.load(file)

# Clean the text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

questions = [clean_text(pair['question']) for pair in data]
answers = [clean_text(pair['answer']) for pair in data]

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

def tokenize(texts):
    return tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=128,  # Adjusted for typical BERT input lengths
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

tokenized_questions = tokenize(questions)

# Create integer labels for answers
answer_ids = list(range(len(answers)))  # Assuming each answer is unique and corresponds to a unique label

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': tokenized_questions['input_ids'],
        'attention_mask': tokenized_questions['attention_mask']
    },
    answer_ids
)).shuffle(len(questions)).batch(8)

# Load BERT model with a suitable configuration
if not os.path.exists('./saved_model'):
    config = BertConfig.from_pretrained('bert-large-uncased', num_labels=len(answer_ids))
    model = TFBertForSequenceClassification.from_pretrained('bert-large-uncased', config=config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    # Compile and train the model within a strategy scope
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(train_dataset, epochs=3)
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')
else:
    model = TFBertForSequenceClassification.from_pretrained('./saved_model')
    tokenizer = BertTokenizer.from_pretrained('./saved_model')

def get_answer(question):
    cleaned_question = clean_text(question)
    inputs = tokenizer.encode_plus(cleaned_question, return_tensors='tf', max_length=128, truncation=True, padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = model(input_ids, attention_mask=attention_mask)
    answer_id = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    answer = answers[answer_id]  # Retrieve the answer using the predicted label
    return answer

# Streamlit interface
st.title("Agriculture Empowerment Chatbot")
st.write("Ask a question related to farming and technology!")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        answer = get_answer(question)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please enter a question.")

if st.button("Clear"):
    question = ""