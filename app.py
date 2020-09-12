from flask import Flask, request
from utils import Model

app = Flask(__name__)

@app.route('/test')
def test():
    return 'Hello World'


@app.route('/', methods = ['POST'])
def generate_lyrics():
    # try:
    data = request.get_json()
    usr_input = data.get('text')
    artist = data.get('artist')
    n_words=50
    seq_length = 50
    generated_lyrics = Model.generate_seq(artist, usr_input, seq_length, n_words)
    return { 'generated_lyrics' : generated_lyrics }
    # except:
    #     return { 'message' : 'Error' }, 500


if (__name__ == '__main__'):
    app.run()
