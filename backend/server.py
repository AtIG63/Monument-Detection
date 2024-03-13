from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Importing deps for image prediction
'''from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    file.save('uploads/' + file.filename)

    # Load the image to predict
    img_path = f"./uploads/{file.filename}"
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255

    loaded_model = load_model('./model/monument_classifier.h5')

    # Make the prediction
    predictions = loaded_model.predict(x)
    if os.path.exists(f"./uploads/{file.filename}"):
        os.remove(f"uploads/{file.filename}")
    
    class_names = ['India Gate', 'Gateway Of India', 'Taj Mahal']  # Replace with your class names
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    #print("Predicted class:", predicted_class)
    return jsonify({"message": predicted_class})


if __name__ == '__main__':
    app.run(debug=True)