# Agriculture Empowerment Chatbot

This repository contains a chatbot designed to answer questions related to empowering farmers with technology for sustainable year-round farming, enhancing food security. The chatbot uses the BERT model fine-tuned on a custom dataset.

## Performance Metrics

The chatbot was evaluated using the following performance metrics:

- **Accuracy**: The accuracy of the chatbot in providing correct responses based on the training dataset.
- **Loss**: The loss function used during training to optimize the model.

Performance results after training for 3 epochs:

- **Accuracy**: 76%
- **Loss**: 0.05

These metrics indicate the chatbot's high accuracy and low error rate in generating responses relevant to the questions asked.

## Installation

To run the chatbot locally, follow these steps:

1. **Clone the Repository**

```bash
git clone https://github.com/vassa33/alu-machine_learning/tree/main
cd chatbot
```

2. **Install Dependencies**

Ensure you have Python 3.7+ and install the necessary libraries:

```bash
pip install tensorflow tensorflow-hub transformers keras streamlit
```

3. **Download the Dataset**

Download the dataset and place it in the same directory as your script. [Download dataset](dataset.json)

4. **Run the Chatbot**

Execute the chatbot script:

```bash
streamlit run chatbot.py
```

## Usage

After running the chatbot script, you can start asking questions related to farming and technology. The chatbot will provide responses based on the trained model.

### Example Conversations
```
**User:** How can technology improve farming?
**Chatbot:** "Farmers can use data analytics to monitor crop health, predict yields, manage resources efficiently, and make informed decisions."
```
```
**User:** "How does climate change impact farming?"
**Chatbot:** "Climate change affects farming by altering weather patterns, increasing the frequency of extreme events, and impacting crop productivity."
```

![image](https://github.com/user-attachments/assets/56ce0fed-3ffb-49bf-b15b-66af6e32d14c)


## Contributing

Contributions are welcome! If you have suggestions for improvements, please create an issue or submit a pull request.


---

Feel free to reach out with any questions or feedback! Happy farming!

---
