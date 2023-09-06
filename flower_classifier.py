import argparse
import numpy as np
import tensorflow as tf
model_path="/Users/harinibennuri/Downloads/Blossom_Detector.ipynb"
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_flower(model, input_features):

    input_features = np.array(input_features).reshape(1, -1)
    predictions = model.predict(input_features)
    predicted_class = np.argmax(predictions)
    return predicted_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flower Species Classifier')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--features', type=float, nargs='+', required=True, help='Input features for classification')
    args = parser.parse_args()

    model = load_model(args.model_path)
    predicted_class = predict_flower(model, args.features)
    print(f'Predicted flower species class: {predicted_class}')
