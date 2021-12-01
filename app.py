
import os, logging
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory,render_template
from werkzeug.utils import secure_filename
import json
import base64
import sys
import os
import cv2
import numpy as np
import predict_web_NAPSOSI_tf2 as predict
from datetime import datetime
from util import base64_to_pil, np_to_base64,base64_to_pts
from PIL import Image
from flask_cors import CORS

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath
    
app = Flask(__name__)
CORS(app)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)

model=predict.PredictionModel()
graph = tf.Graph()
with graph.as_default():
    session = tf.compat.v1.Session()
    with session.as_default():
        model.create_model()


@app.route('/api_predict', methods=['POST'])
def api_predict_torus():
    img = np.array(base64_to_pil(request.json))
    #name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")+'_'+".jpeg"
    #path = "uploads/" + name
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(path,img)
    
    with graph.as_default():
        with session.as_default():
           list_img_nail_separate,list_8class,list_NAPSOSI,img0_text = model.NAPSOSI_score_and_nail_separate(img)
           results  = []
           for i in range(len(list_NAPSOSI)):
               results.append({"img":np_to_base64(list_img_nail_separate[i][:, :, ::-1]),
                               "class_score":list_8class[i],
                               "NAPSOSI_score":list_NAPSOSI[i]})
           
    return jsonify(results=results,img_with_numerate=np_to_base64(img0_text[:, :, ::-1]))

if __name__ == '__main__':    
    #start server 

    app.run(host="0.0.0.0", port=8889)
    
