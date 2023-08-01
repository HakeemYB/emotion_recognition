from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import base64

app = Flask(__name__)

# Load the trained model
file_name = 'best_model.h5'
checkpoint_path = os.path.join('checkpoint', file_name)
model = load_model(checkpoint_path)

def preprocess_image(image):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize the image to the required input size of the model
    input_shape = model.input_shape[1:3]  # Assuming the model's input shape is (height, width, channels)
    resized_image = cv2.resize(image_gray, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)

    # Convert the resized image to a 4D tensor to match the model's input shape
    input_tensor = np.expand_dims(resized_image, axis=-1)

    # Normalize pixel values to the range [0, 1]
    input_tensor = input_tensor.astype('float32') / 255.0

    return input_tensor

@app.route('/', methods=['GET', 'POST'])
def predict_expression():
    if request.method == 'POST':
        # Get the user's uploaded image
        image_file = request.files['image']
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Make the prediction using the loaded model
        prediction = model.predict(input_tensor[np.newaxis, ...])[0]
        predicted_expression = np.argmax(prediction)

        # Define a list of facial expressions (modify as needed based on your model's output classes)
        facial_expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

         # Convert the image to base64 string
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')


        # Get the predicted facial expression
        predicted_expression_label = facial_expressions[predicted_expression]

        return render_template('result.html', predicted_expression=predicted_expression_label, image_base64=image_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
