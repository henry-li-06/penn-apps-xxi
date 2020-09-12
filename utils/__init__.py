import numpy as np
import pandas as pd
from keras.models import load_model
import re
from unidecode import unidecode
from random import randint

model = load_model('./word_model_travis_scott.h5')


def get_tokenized_lines(df):
    words = []
    
    for index, row in df['lyrics'].iteritems():
        row = str(row).lower()
        for line in row.split('|-|'):
            new_words = re.findall(r"\b[a-z']+\b", unidecode(line))
            words = words + new_words
        
    return words

songs = pd.read_csv('./data/travis-scott-lyrics.csv')
all_lyric_lines = get_tokenized_lines(songs)
vocab = set(all_lyric_lines)
vocab = sorted(vocab)
word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
word_indices = [word_to_index[word] for word in vocab]
vocab_size = len(vocab)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def texts_to_sequences(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)
    
    for i, text in enumerate(texts):
        if text not in word_to_index:
            random = index_to_word[randint(0,vocab_size)]
            indices[:, i] = word_to_index[random]
        else:
            indices[:, i] = word_to_index[text]
        
    return indices

def my_pad_sequences(seq, maxlen):
    start = seq.shape[1] - maxlen
    return seq[:, start: start + maxlen]

def generate_seq(usr_input, seq_length, n_words):
    generated = ''
    
    result = list()
    in_text = [None] * 51
    generated_list = generated.split()
    
    # if input is shorter than 51 words, fill the beginning with random words
    if(len(generated_list) < 51):
        end = len(generated_list)
        for i in range (51 - end):
            random = index_to_word[randint(0,vocab_size)]
            in_text[i] = random
            
        index = 0
        for i in range (51 - end, 51):
            in_text[i] = generated_list[index]
            index += 1

    # if input is longer than 51 words, only use the last 51 words
    if(len(generated_list) > 51):
        end = len(generated_list)
        in_text = generated_list[end-51:]

    # generate words based on last 50 words
    for _ in range(n_words):
        encoded = texts_to_sequences(in_text[1:], word_to_index)
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