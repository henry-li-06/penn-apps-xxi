from flask import Flask, request
from utils import *

app = Flask(__name__)

songs = pd.read_csv('data/eminem-lyrics.csv')

@app.route('/test')
def test():
    return 'Hello World'


@app.route('/', methods = ['POST'])
def generate_lyrics():
    # try:
    data = request.get_json()
    usr_input = data.get('text')
    n_words=50
    seq_length = 50
    generated_lyrics = generate_seq(usr_input, seq_length, n_words)
    return { 'generated_lyrics' : generated_lyrics }
    # except:
    #     return { 'message' : 'Error' }, 500


if (__name__ == '__main__'):
    app.run()
