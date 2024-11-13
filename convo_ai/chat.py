import random
import json
import torch
from model import NeuralNet
from utils import tokenize, bag_of_words
from flask import Flask, request, render_template
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
path = os.getcwd()

with open('intents.json', 'r') as intent_data:
    intents = json.load(intent_data)

bot_name = "ChatBot"
model_path = "model.pth"

data = torch.load(model_path)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def get_response(tag):
    for intent in intents["intents"]:
        if tag in intent["tag"]:
            response = random.choice(intent["responses"])
            suggestion = intent["suggestion"] if "suggestion" in intent.keys() else ""
    return response, suggestion

@app.route('/', methods=["GET"])
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=["POST"])
def chat():
    data = request.get_json()
    sentence = data.get('message', '')
    if len(sentence.split()) == 1:  # If the input is short (e.g., 'hii')
        threshold = 0.5
    else:
        threshold = 0.65
    tokenize_sentence = tokenize(sentence)
    bow = bag_of_words(tokenize_sentence, all_words)
    bow = torch.from_numpy(bow).float().unsqueeze(0)
    output = model(bow)
    probs = torch.softmax(output, dim=1)
    max_prob, predicted_index = torch.max(probs, dim=1)
    predicted_tag = tags[predicted_index.item()]
    confidence = max_prob.item()
    
    if confidence > threshold:
        response, suggestion = get_response(predicted_tag)
        print(f"{bot_name}: {response}")
        return {'response': response, 'suggestion': suggestion}
    else:
        intent = intents["intents"][-1]
        response = random.choice(intent["responses"])
        print(f"{bot_name}: Sorry, i don't understand")
        return {'response': response, 'suggestion': intent["suggestion"] if "suggestion" in intent.keys() else ""}


if __name__ == "__main__":
    import nltk
    nltk.download('punkt_tab')
    nltk.download('wordnet')  
    app.run(host='0.0.0.0', port=5001, debug=True)