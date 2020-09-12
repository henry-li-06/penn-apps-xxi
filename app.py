from flask import Flask, request
from keras.models import load_model
import numpy as np
import sys

app = Flask(__name__)


@app.route('/')
def get_predicted_lyrics():
    try:
        usr_input = request
        model = load_model('char_model_eminem.h5')
        n_words=50
        seq_length = 50

        generate_seq(model, word_to_index, seq_length, usr_input, n_words)
    except:
        return 'This is no good'


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# def generate_output(model, Tx, chars, usr_input, char_indices, indices_char):
#     generated = ''
#     usr_input = usr_input
#
#     sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
#     generated += usr_input
#
#     sys.stdout.write("\n\nHere is your poem: \n\n")
#     sys.stdout.write(usr_input)
#     for i in range(400):
#
#         x_pred = np.zeros((1, Tx, len(chars)))
#
#         for t, char in enumerate(sentence):
#             if char != '0':
#                 x_pred[0, t, char_indices[char]] = 1.
#
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, temperature=0.2)
#         next_char = indices_char[next_index]
#
#         generated += next_char
#         sentence = sentence[1:] + next_char
#
#         sys.stdout.write(next_char)
#         sys.stdout.flush()
#
#         if next_char == '\n':
#             continue

vocab = set(all_lyric_lines)
word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
word_indices = [word_to_index[word] for word in vocab]
vocab_size = len(vocab)


def texts_to_sequences(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)

    for i, text in enumerate(texts):
        indices[:, i] = word_to_index[text]

    return indices


def my_pad_sequences(seq, maxlen):
    start = seq.shape[1] - maxlen
    return seq[:, start: start + maxlen]


def generate_seq(model, word_to_index, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text

    for _ in range(n_words):
        encoded = texts_to_sequences(in_text.split()[1:], word_to_index)
        encoded = my_pad_sequences(encoded, maxlen=seq_length)

        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''

        for word, index in word_to_index.items():
            if index == yhat:
                out_word = word
                break

        in_text += ' ' + out_word
        result.append(out_word)

    return ' '.join(result)


if (__name__ == '__main__'):
    app.run()
