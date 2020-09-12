import numpy as np
import pandas as pd
from keras.models import load_model
import re
from unidecode import unidecode
from random import randint


class Model:
    
    @classmethod
    def set_artist(cls, artist):
        cls.model = load_model('./word_model_{}.h5'.format(artist))
        songs = pd.read_csv('./data/{}-lyrics.csv'.format(artist.replace('_', '-')))
        all_lyric_lines = Model.get_tokenized_lines(songs)
        vocab = set(all_lyric_lines)
        vocab = sorted(vocab)
        cls.word_to_index = {w: i for i, w in enumerate(vocab)}
        cls.index_to_word = {i: w for w, i in cls.word_to_index.items()}
        cls.vocab_size = len(vocab)

    @staticmethod
    def get_tokenized_lines(df):
        words = []
        
        for index, row in df['lyrics'].iteritems():
            row = str(row).lower()
            for line in row.split('|-|'):
                new_words = re.findall(r"\b[a-z']+\b", unidecode(line))
                words = words + new_words
            
        return words

    @classmethod
    def texts_to_sequences(cls, texts):
        indices = np.zeros((1, len(texts)), dtype=int)
        
        for i, text in enumerate(texts):
            if text not in cls.word_to_index:
                random = cls.index_to_word[randint(0,cls.vocab_size)]
                indices[:, i] = cls.word_to_index[random]
            else:
                indices[:, i] = cls.word_to_index[text]
            
        return indices

    @staticmethod
    def my_pad_sequences(seq, maxlen):
        start = seq.shape[1] - maxlen
        return seq[:, start: start + maxlen]

    @classmethod
    def generate_seq(cls, artist, usr_input, seq_length, n_words):
        cls.set_artist(artist)
        generated = ''
        
        result = list()
        in_text = [None] * 51
        generated_list = generated.split()
        
        # if input is shorter than 51 words, fill the beginning with random words
        if(len(generated_list) < 51):
            end = len(generated_list)
            for i in range (51 - end):
                random = cls.index_to_word[randint(0,cls.vocab_size)]
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
            encoded = cls.texts_to_sequences(in_text[1:])
            encoded = Model.my_pad_sequences(encoded, maxlen=seq_length)
            
            yhat = cls.model.predict_classes(encoded, verbose=0)
            out_word = ''
        
            for word, index in cls.word_to_index.items():
                if index == yhat:
                    out_word = word
                    break
            
            in_text += ' ' + out_word
            result.append(out_word)
            
        return ' '.join(result)