"""
Flask application to run recycle model
"""
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import numpy as np
import joblib
import cv2
import traceback
import os
from pathlib import Path
from io import BytesIO
from fastai.vision import *

clear_session()
app = Flask(__name__, template_folder="templates")
recycle_model = None

material_class = [ 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash' ]

def load_recycle_model():
    global recycle_model
    path = Path(os.getcwd())
    
    ######## h5 keras model load ########

    # model_file = './keras-recycle.h5'
    # Loads from h5 file
    # recycle_model = load_model(model_file)
    
    # build and compile on GPU
    # recycle_model._make_predict_function()

    #####################################

    ###### Fastai model load ############

    model_file = './recycle-fastai.pkl'
    recycle_model = load_learner(path, file = 'recycle-fastai.pkl', bs=16, num_workers = 0 )
     
    #####################################
    
@app.route("/recycle", methods=["POST"])
def recycle_predict():
    try:
        image =  request.files['file'].read()
     
        ###### Keras model predict ###############

        # print("Got image")
        # # https://stackoverflow.com/a/27537664/818687
        # arr = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_GRAYSCALE)
        # print("CV2 read image")
        # my_image = arr / 255.0
        # my_images = my_image.reshape(-1, 384, 512, 1)
        # print("Got here")

        # pred = recycle_model.predict(my_images)
        # n = int(np.argmax(pred))
        # print(n)
      
        ############################################
      
    
        ########## Fastai model predict ############

        img = open_image(BytesIO(image))
        n, pred_idx, outputs = recycle_model.predict(img)
        print(n, pred_idx)

        ############################################
      
        return jsonify({
            "message": f"The predicted object for the uploaded image is {str(n)} ",
            "class": str(n)
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