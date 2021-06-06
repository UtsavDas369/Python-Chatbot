
#importing natural language toolit
import nltk
nltk.download('punkt')
nltk.download('wordnet')
#Sentence Tokenizer == punkt,wordnet == database for english language
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()  #converting a word to its base form


import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
#Initializing variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
#loading intents.json for training the data
data_file = open('intents.json').read()
intents = json.loads(data_file)

#Documenting all the words/making a list of all the words and lemmatized words which helps the tensorflow to train the data and come out with the reply of the questions asked
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #adding documents
        documents.append((w, intent['tag']))

        #adding classes to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatizing each word, in other words converting the wordings to base form
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

#printing the number of documents,classes also priniting out the base wordings for reference
print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

#serializing the wordings file will be generated, we can see this inside the project
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#initializing training list
training = []
output_empty = [0] * len(classes)
for doc in documents:
    #from documents listing all the tokenized words for the patternand in attempt to represent related words if the word match is found we print the answer , 
    # in other words if the question has that word we find which answer has the same word and print it
    bag = []
  
    pattern_words = doc[0]
 
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
   
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #output=0 for each tag, current tag=1 (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
#random shuffling putting in numpy array
random.shuffle(training)
training = np.array(training)
#x = intents y =patterns training the data
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#Creating the model of 3 layers 1st layer =128 nuerons 2nd = 64 3rd = no of intents

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
#we use softmax to predict outputs of intents

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
