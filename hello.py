import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Parameters
fixed_length = 1000
lowcut, highcut = 8, 30
data_dir = "dataBCI"
temp_dir = "temp_data"
os.makedirs(temp_dir, exist_ok=True)
results = {}

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Preprocessing function
def preprocess_trials(session, fs):
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

# Save subject data incrementally to temp dir
all_labels = []
for subject_id in range(1, 10):
    train_file = os.path.join(data_dir, f"A0{subject_id}T.mat")
    try:
        train_mat = loadmat(train_file)
        session_train = train_mat['data'][0][3]
        fs = session_train['fs'][0][0][0][0]

        X, y = preprocess_trials(session_train, fs)
        X = X.reshape(X.shape[0], -1)

        # Save to .npy for later combined loading
        np.save(os.path.join(temp_dir, f"X_sub{subject_id}.npy"), X)
        np.save(os.path.join(temp_dir, f"y_sub{subject_id}.npy"), y)
        print(f"✅ Processed A0{subject_id}T.mat")
    except Exception as e:
        print(f"❌ Failed A0{subject_id}T.mat: {e}")

# Combine preprocessed features and labels from disk
X_combined = []
y_combined = []
for subject_id in range(1, 10):
    try:
        X = np.load(os.path.join(temp_dir, f"X_sub{subject_id}.npy"))
        y = np.load(os.path.join(temp_dir, f"y_sub{subject_id}.npy"))
        X_combined.append(X)
        y_combined.append(y)
    except Exception as e:
        print(f"⚠️ Skipped A0{subject_id}: {e}")

X_all = np.vstack(X_combined)
y_all = np.hstack(y_combined)

# Label encoding
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y_all)

# Train generalized model
print("\n🚀 Training SVM on all subjects...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_all, y_enc)

# Save model and encoder
os.makedirs("saved_models", exist_ok=True)
joblib.dump(svm, "saved_models/generalized_model.pkl")
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")

print("✅ Generalized model saved!")

# Evaluate model on each subject's test set
print("\n📊 Evaluation per subject:")
for subject_id in range(1, 10):
    test_file = os.path.join(data_dir, f"A0{subject_id}E.mat")
    try:
        test_mat = loadmat(test_file)
        session_test = test_mat['data'][0][3]
        X_test, y_test = preprocess_trials(session_test, fs)
        X_test = X_test.reshape(X_test.shape[0], -1)
        y_test_enc = label_encoder.transform(y_test)

        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        report = classification_report(y_test_enc, y_pred, output_dict=True)
        results[f"A0{subject_id}"] = {"accuracy": acc, "report": report}

        print(f"\n✅ A0{subject_id} Accuracy: {acc:.2f}")
    except Exception as e:
        print(f"❌ A0{subject_id}E.mat error: {e}")
