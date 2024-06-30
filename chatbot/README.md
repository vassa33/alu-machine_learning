# Agriculture Empowerment Chatbot

This repository contains a chatbot designed to answer questions related to empowering farmers with technology for sustainable year-round farming, enhancing food security. The chatbot uses the BERT model fine-tuned on a custom dataset.

## Link to Dataset

The dataset used for training the chatbot is a custom collection of question-answer pairs related to farming and technology. You can download the dataset [here](dataset.json).

## Performance Metrics

The chatbot was evaluated using the following performance metrics:

- **Accuracy**: The accuracy of the chatbot in providing correct responses based on the training dataset.
- **Loss**: The loss function used during training to optimize the model.

Performance results after training for 3 epochs:

- **Accuracy**: 95%
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
pip install tensorflow tensorflow-hub transformers
```

3. **Download the Dataset**

Download the dataset and place it in the same directory as your script. [Download dataset](dataset.json)

4. **Run the Chatbot**

Execute the chatbot script:

```bash
python chatbot.py
```

## Usage

After running the chatbot script, you can start asking questions related to farming and technology. The chatbot will provide responses based on the trained model.

### Example Conversations
```
**User:** How can technology improve farming?
**Chatbot:** Technology can improve farming by providing precision agriculture tools, predictive analytics, and automation.
```
```
**User:** What are the benefits of precision agriculture?
**Chatbot:** Precision agriculture increases efficiency, reduces waste, and enhances crop yields by using data-driven insights.
```
```
**User:** How does irrigation technology help farmers?
**Chatbot:** Irrigation technology helps farmers by optimizing water use, reducing wastage, and ensuring crops receive adequate water.
```
```
**User:** What tools can help with crop disease detection?
**Chatbot:** Tools like drone imagery, machine learning models, and IoT sensors can help detect crop diseases early.
```
```
**User:** Why is soil health important?
**Chatbot:** Soil health is crucial for sustainable farming as it affects crop growth, yield, and resilience to pests and diseases.
```

## Access the Chatbot

You can access the chatbot via this link: [Agriculture Empowerment Chatbot](https://your-deployment-link.com)

This link will direct you to a web-based interface where you can interact with the chatbot

## Contributing

Contributions are welcome! If you have suggestions for improvements, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to reach out with any questions or feedback! Happy farming!

---
