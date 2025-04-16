import os
from flask import Flask, request, jsonify
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
import numpy as np
import joblib
import traceback
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

# CORS(app)  # Add this line


fixed_length = 1000
lowcut, highcut = 8, 30
model = None

label_map = {
    1: 'Left Hand',
    2: 'Right Hand',
    3: 'Feet',
    4: 'Tongue'
}

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def preprocess_trials(session, fs):
    print("✅ Preprocessing trials...")
    raw_data = session['X'][0][0]
    trial_starts = session['trial'][0][0]
    labels = session['y'][0][0]

    trials = []
    trial_labels = []
    for i in range(len(trial_starts)):
        start = (trial_starts[i][0] - 1)
        end = (trial_starts[i + 1][0] - 1) if i < len(trial_starts) - 1 else raw_data.shape[0]
        trial = raw_data[start:end, :]

        if trial.shape[0] >= fixed_length:
            trimmed = trial[:fixed_length, :]
        else:
            padded = np.zeros((fixed_length, raw_data.shape[1]))
            padded[:trial.shape[0], :] = trial
            trimmed = padded

        trials.append(trimmed)
        trial_labels.append(labels[i][0])

    trials = np.array([zscore(trial, axis=0) for trial in trials])
    filtered = np.array([bandpass_filter(trial, lowcut, highcut, fs) for trial in trials])
    return filtered, np.array(trial_labels)

def load_model():
    global model
    if model is None:
        print("🔄 Loading model...")
        model = joblib.load('saved_models/generalized_model.pkl')
    return model

@app.route('/')
def index():
    return "Flask is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 File received!")
        if 'file' not in request.files:
            print("🚫 No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            print("🚫 No file selected")
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.mat'):
            print("📂 Loading .mat file...")
            test_file = loadmat(BytesIO(file.read()))

            session_test = test_file['data'][0][3]
            fs = session_test['fs'][0][0][0][0]
            print("✅ Data loaded successfully!")

            X_test, y_test = preprocess_trials(session_test, fs)
            X_test = X_test.reshape(X_test.shape[0], -1)

            model = load_model()
            print("⚙️ Predicting...")
            y_pred = model.predict(X_test)

            predictions = []
            for i in range(len(y_pred)):
                pred_label = label_map.get(y_pred[i], f"Unknown ({y_pred[i]})")
                true_label = label_map.get(y_test[i], f"Unknown ({y_test[i]})")
                predictions.append(f"True: {true_label}, Predicted: {pred_label}")

            print("✅ Prediction done!")
            return jsonify({"success": True, "predictions": predictions})

        else:
            print("🚫 Invalid file type")
            return jsonify({"error": "Invalid file type. Please upload a .mat file."}), 400

    except Exception as e:
        print("🔥 Exception occurred:")
        traceback.print_exc()
        return jsonify({"error": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
