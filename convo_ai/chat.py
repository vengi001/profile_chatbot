
import random
import json
import torch
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
from model import NeuralNet
from utils import tokenize, bag_of_words

app = Flask(__name__)
CORS(app)

# Lazy-loaded components
model = None
all_words = []
tags = []
intents = None
model_path = "model.pth"
intents_path = "intents.json"

bot_name = "ChatBot"

def load_intents():
    """Lazy-load intents file."""
    global intents
    if intents is None:
        with open(intents_path, 'r') as intent_data:
            intents = json.load(intent_data)

def load_model():
    """Lazy-load the model to save memory."""
    global model, all_words, tags
    if model is None:
        data = torch.load(model_path, map_location=torch.device('cpu'))  # Load on CPU
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
    """Fetch response and suggestions for the given tag."""
    load_intents()
    for intent in intents["intents"]:
        if tag == intent["tag"]:
            response = random.choice(intent["responses"])
            suggestion = intent.get("suggestion", "")
            return response, suggestion
    return "I'm sorry, I don't understand.", ""

@app.route('/', methods=["GET"])
def home():
    """Render the chatbot UI."""
    return render_template('chatbot.html')

@app.route('/chat', methods=["POST"])
def chat():
    """Handle chat requests."""
    load_model()  # Ensure model is loaded only when needed
    load_intents()  # Ensure intents are loaded only when needed

    data = request.get_json()
    sentence = data.get('message', '')

    # Adjust confidence threshold based on sentence length
    threshold = 0.5 if len(sentence.split()) == 1 else 0.65

    # Tokenize and predict
    tokenized_sentence = tokenize(sentence)
    bow = bag_of_words(tokenized_sentence, all_words).astype(np.float16)  # Use smaller data type
    bow = torch.from_numpy(bow).float().unsqueeze(0)
    output = model(bow)
    probs = torch.softmax(output, dim=1)
    max_prob, predicted_index = torch.max(probs, dim=1)
    predicted_tag = tags[predicted_index.item()]
    confidence = max_prob.item()

    if confidence > threshold:
        response, suggestion = get_response(predicted_tag)
        return {'response': response, 'suggestion': suggestion}
    else:
        fallback = intents["intents"][-1]
        response = random.choice(fallback["responses"])
        return {'response': response, 'suggestion': fallback.get("suggestion", "")}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)
