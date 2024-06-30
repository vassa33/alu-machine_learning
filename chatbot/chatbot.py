import json
import re
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig


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
tokenized_answers = tokenize(answers)

train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': tokenized_questions['input_ids'],
        'attention_mask': tokenized_questions['attention_mask']
    },
    tokenized_answers['input_ids']
)).shuffle(len(questions)).batch(8)

# Load BERT model with a suitable configuration
config = BertConfig.from_pretrained('bert-large-uncased', num_labels=tokenizer.vocab_size)
model = TFBertForSequenceClassification.from_pretrained('bert-large-uncased', config=config)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(train_dataset, epochs=3)

def get_answer(question):
    cleaned_question = clean_text(question)
    inputs = tokenizer.encode_plus(cleaned_question, return_tensors='tf', max_length=128, truncation=True, padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = model(input_ids, attention_mask=attention_mask)
    answer_start = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    answer_tokens = input_ids[0, answer_start:answer_start+10]  # Adjust slice length as needed
    answer = tokenizer.decode(answer_tokens)
    return answer

def chatbot_interface():
    print("Welcome to the Agriculture Chatbot. Ask a question related to farming and technology!")
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "bye"]:
            print("A: Goodbye!")
            break
        answer = get_answer(question)
        print(f"A: {answer}")

# Run the chatbot interface
if __name__ == "__main__":
    chatbot_interface()
