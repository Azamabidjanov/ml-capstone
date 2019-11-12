"""
Flask application to run recycle model
"""
from flask import Flask, jsonify, request, render_template
from keras.models import load_model
from keras.backend import clear_session
import numpy as np
import joblib
import cv2
import traceback
import os
from pathlib import Path

clear_session()
app = Flask(__name__)
recycle_model = None

def load_recycle_model():
    global recycle_model
    #path = Path(os.getcwd())/"data"
    #model_file = f'{path}/models/keras-recycle.pth'
    model_file = 'keras-mnist.h5'
    # Loads from h5 file
    recycle_model = load_model(model_file)
    # build and compile on GPU
    recycle_model._make_predict_function()

@app.route("/recycle", methods=["POST"])
def recycle_predict():
    try:
        image =  request.files['file'].read()
        print("Got image")
        # https://stackoverflow.com/a/27537664/818687
        arr = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)
        print("CV2 read image")
        my_image = arr / 255.0
        my_images = my_image.reshape(1, 28, 28, 1)
        print("Got here")

        pred = recycle_model.predict(my_images)
        n = int(np.argmax(pred))
        print(n)
        return jsonify({
            "message": f"The predicted object for the uploaded image is {n} ",
            "number": n
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "message": f"An error occurred. {e}"
        })

@app.route("/recycle-ui")
def recycle_ui():
    return render_template("recycle.html")


if __name__ == '__main__':
    load_recycle_model()
    app.run(debug=True)