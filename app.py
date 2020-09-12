from flask import Flask, request
from utils import Model

app = Flask(__name__)

MAX_ATTEMPTS = 10

@app.route('/', methods = ['POST'])
def generate_lyrics():
    try:
        data = request.get_json()
        usr_input = data.get('text')
        artist = data.get('rapper')
        for _ in range(MAX_ATTEMPTS):
            try:
                generated_lyrics = Model.generate_seq(artist, usr_input)
                return { 'generated_lyrics' : generated_lyrics }
            except:
                pass
    except:
        return { 'message' : 'There was an error generating lyrics' }, 500


if (__name__ == '__main__'):
    app.run()
