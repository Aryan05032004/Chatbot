import nltk
from nltk.stem.lancaster import LancasterStemmer
Stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import json
import random
import pickle

with open("json file/intents.json") as file:
    data = json.load(file)

try:
    with open("list_data" , "rb") as f:
        words , labels , training , output = f
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])


        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [Stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [Stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("list_data" , "wb") as f:
        pickle.dump((words , labels , training , output) , f)


# tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0]) ])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    sen_words = nltk.word_tokenize(s)
    sen_words = [Stemmer.stem(word.lower()) for word in sen_words]

    for i in sen_words:
        for j, w in enumerate(words):
            if w == i:
                bag[j] = 1
 
    return np.array(bag)


def chat():
    print("Bot : Speak idiot")
    while(True):
        inp = input("You : ")

        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])
        result_index = np.argmax(result)
        tag = labels[result_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                response = tg['responses']

        print("Bot : ", random.choice(response))
        
    
    
chat()
