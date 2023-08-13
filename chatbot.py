import random
import json
import pickle
import numpy as np
import tensorflow as tf
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download the wordnet resource before using the NLTK lemmatizer
nltk.download('wordnet')

# Import the required NLTK module
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, user_name):
    max_probability = 0
    best_intent = None

    for intent_item in intents_list:
        intent_tag = intent_item['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                response = response.replace("{name}", user_name)
                probability = float(intent_item['probability'])

                if probability > max_probability:
                    max_probability = probability
                    best_intent = response

    if best_intent:
        return best_intent
    else:
        return "I'm sorry, I don't have a suitable response for that."

print("GO! BOT IS RUNNING")
user_name = input("Bot: Hi there, ! What's your name? ")
while True:
    message = input(user_name + ": ")
    if message.lower() == "exit":
        print("Bot: Goodbye, " + user_name + "! Have a great day.")
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents, user_name)
    print("Bot:", response)
