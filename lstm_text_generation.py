from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, LSTM, Bidirectional
from keras.optimizers import *
from keras.utils.data_utils import get_file
from keras import backend as K

import numpy as np
import random
import math
import sys
import io
from string import ascii_lowercase
from json import load
import re
from gc import collect
from datetime import datetime

# ================================================
# Configuration
# ================================================

max_len              = 40
step                 = 3
permitted_characters = ascii_lowercase + " '?!"
joining_characters   = " | "
lstm_size            = 512
number_of_layers     = 3
learning_rate        = 0.001
clipping_value       = 0.5
batch_size           = 128
epochs               = 100
bidirectional_lstm   = True
batch_normalization  = True

# ================================================
# ================================================

def output(*args):
    print("[{}]".format(datetime.now().isoformat()), *args)

class Joke:
    def __init__(self, json):
        self.id, self.title, self.body, self.score = json["id"], json["title"], json["body"], json["score"]

    def transform(self):
        string = (self.title.lower() + " " + self.body.lower()).strip().replace("’", "'")
        new_string = ""
        for s in string:
            if s in permitted_characters: new_string += s
            else: new_string += " "
        new_string = re.sub(' +', ' ', new_string).strip()
        return new_string


path = "/media/khaled/thor/Repositories/openai-jokes-dataset/reddit_jokes.json"
json_jokes = [Joke(x).transform() for x in load(open(path, 'r'))]
text = joining_characters.join(json_jokes)
text = text[:int(2.5 * 1e6)]

del json_jokes
collect()

chars = sorted(list(set(text)))

output('Corpus length:', len(text))
output('Total chars:',   len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

sentences   = []
next_chars  = []

output('Building the model.')

model = Sequential()

model.add(BatchNormalization(input_shape=(max_len, len(chars))))

for i in range(number_of_layers - 1):
    if bidirectional_lstm:
        model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
    else:
        model.add(LSTM(lstm_size, return_sequences=True))

    if batch_normalization:
        model.add(BatchNormalization())

if bidirectional_lstm:
    model.add(Bidirectional(LSTM(lstm_size, return_sequences=False)))
else:
    model.add(LSTM(lstm_size, return_sequences=False))

if batch_normalization:
    model.add(BatchNormalization())

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=learning_rate, clipvalue=clipping_value)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

output('Number of sequences:', len(sentences))
output('Started vectorization.')

collect()

x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

collect()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    print("\n")
    output('----- Generating text after epoch: {}'.format(epoch))
    start_index = random.randint(0, len(text) - max_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        output('----- Diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + max_len]
        generated += sentence
        output('----- Generating with seed: "{}"'.format(sentence))
        sys.stdout.write(generated)
        for i in range(400):
            x_pred = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print("\n")

print_callback = LambdaCallback(
    on_epoch_end=on_epoch_end,
)

model.fit(
    x, y,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[print_callback],
)
