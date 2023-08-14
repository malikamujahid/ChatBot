import random
import json
import pickle
import numpy as np
import tensorflow as tf
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Download the wordnet resource before using the NLTK lemmatizer
nltk.download('wordnet')

# Import the required NLTK module
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
print("Model training started.")
sys.stdout.flush()

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Split data into training and validation sets
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

hist = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping])
model.save('chatbot_model.h5', hist)
print('Model training completed.')
