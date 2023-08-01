# Facial Expression Recognition using Flask and Keras

This project implements a facial expression recognition system using a Convolutional Neural Network (CNN) model trained on the FER2013 dataset. The trained model is deployed as a web application using Flask.

## Prerequisites

Before running the application, you need to have the following installed on your system:

- Python 3
- Flask
- OpenCV
- NumPy
- TensorFlow

You can install the required packages using the following command:

```bash
pip install flask opencv-python numpy tensorflow
```

## Getting Started

1. Clone the repository:
   
```bash
git clone https://github.com/HakeemYB/emotion_recognition.git
cd emotion_recognition
```
2. Download the FER2013 dataset and place it in the root folder of the project. The dataset can be downloaded from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

3. Train the model and save it using `expression_detec.ipynb` notebook.

4. Start the Flask application:

```bash
python3 app.py
```

5. Open your web browser and navigate to `http://localhost:8000`. You should see a simple web interface to upload an image.

## How it Works

- The Flask app provides a simple HTML form to upload an image.
- The uploaded image is sent to the server for processing.
- The image is preprocessed to match the input shape of the trained CNN model (48x48 grayscale).
- The preprocessed image is then fed to the CNN model for facial expression prediction.
- The predicted facial expression is displayed on the result page along with the processed image.

## Folder Structure

- `app.py`: The main Flask application.
- `expression_detec.ipynb`: Jupyter notebook to train the CNN model and save it.
- `templates/`: Contains the HTML templates for the web application.
- `checkpoint/`: Contains saved model for predictions.

## Acknowledgments

This project uses the FER2013 dataset for training the model. The dataset can be found [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
