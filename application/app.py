from flask import Flask, render_template, jsonify, request
import threading
from PIL import Image
import os
import numpy as np
import ctypes
import io

mingw_bin = "C:/msys64/mingw64/bin"
os.add_dll_directory(mingw_bin)

classnames_array = ['buffalo', 'elephant', 'zebre']
mlp = ctypes.cdll.LoadLibrary("./cmake-build-debug/libmlp.dll") 
mlp.release_mlp_model.argtypes = [ctypes.c_void_p]
mlp.load_mlp_model.argtypes = [ctypes.c_char_p]
mlp.load_mlp_model.restype = ctypes.c_void_p
mlp.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_float)
mlp.predict_mlp_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]

model = mlp.load_mlp_model(b"./mlp_trained.bin")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    image_file = request.files.get('image')
    
    if not image_file:
        return jsonify({'result': 'Aucune image reçue'}), 400

    try:
        img = Image.open(image_file.stream)
        
    except Exception as e:
        return jsonify({'result': f"Erreur lecture image : {e}"}), 500

    img = img.convert("RGB")
    img = img.resize(size=(32,32))
    img_data = np.array(img) / 255.0
    output_array = mlp.predict_mlp_model(model, img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    output_array = ctypes.cast(output_array, ctypes.POINTER(ctypes.c_float * len(classnames_array))).contents
    output = list(output_array)
    class_index = 0
    for i in range(len(output)):
        if output[i] == max(output):
            class_index = i
    
    return jsonify({'result': classnames_array[class_index]})

if __name__ == '__main__':
    app.run(debug=True)