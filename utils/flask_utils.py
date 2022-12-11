"""
@Project :acou-input-single
@File ：flask_utils.py
@Date ： 2022/12/4 22:58
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from flask import Flask, request
import numpy as np
import os

from utils.plot_utils import show_signals

UPLOAD_FOLDER = r"../audio"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
arr = np.array([])


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/recognition/file', methods=['POST'])
def recognition_file():
    audio = request.files.get("audio", None)
    audio.save(os.path.join(app.config['UPLOAD_FOLDER'], audio.filename))
    return "recognition"


@app.route('/recognition/stream', methods=['POST'])
def recognition_stream():
    stream_bytes = request.files.get("stream", None).read()
    audio_bytes = np.reshape(np.frombuffer(stream_bytes, dtype=np.int16), (-1, 2), order='F')[:, 1]
    # show_signals(audio_bytes)
    # audio.save(os.path.join(app.config['UPLOAD_FOLDER'], audio.filename))
    return "stream"


if __name__ == '__main__':
    # app.run()

    app.run(host='0.0.0.0', port=5000, debug=True)
