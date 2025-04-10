import os
import glob
import streamlit as st
import torch
import torchaudio
import librosa
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.pretrained import SpeakerRecognition

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    A PyTorch-based LSTM classifier for sequence data.

    Attributes:
        input_size (int): The number of features in the input sequence.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of stacked LSTM layers.
        num_classes (int): The number of output classes for classification.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Initializes the LSTMClassifier.

        Args:
            input_size (int): Number of features in the input sequence.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define an LSTM layer
        # input_size: number of input features per time step
        # hidden_size: number of features in the hidden state
        # num_layers: number of stacked LSTM layers
        # batch_first=True means input/output tensors have shape (batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer 1
        # Takes the hidden state from the LSTM and projects it to 1024 dimensions
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 1024),  # Fully connected layer
            nn.ReLU()  # Activation function
        )

        # Fully connected layer 2
        # Maps the 1024-dimensional output to the number of classes
        # Softmax activation ensures outputs represent probabilities
        self.fc2 = nn.Sequential(
            nn.Linear(1024, num_classes),  # Fully connected layer
            nn.Softmax(dim=1)  # Apply Softmax along the class dimension
        )

    def forward(self, x):
        """
        Defines the forward pass of the LSTMClassifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes), representing class probabilities.
        """
        # Initialize hidden and cell states with zeros
        # Shape of h0 and c0: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Cell state

        # Pass the input sequence through the LSTM
        # out: output features from all time steps (batch_size, seq_length, hidden_size)
        # _: hidden and cell states from the last time step
        out, _ = self.lstm(x, (h0, c0))

        # Use only the output from the last time step
        # Shape of out after slicing: (batch_size, hidden_size)
        out = out[:, -1, :]

        # Pass the output through the first fully connected layer and activation
        out = self.fc1(out)

        # Pass the result through the second fully connected layer and apply Softmax
        out = self.fc2(out)

        return out



# -------- DEVICE SETUP --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD MODELS --------
gender_model = torch.load("sound_model.pth", map_location=device, weights_only=False)
gender_model.to(device)
gender_model.eval()

emotion_model = load_model('New_emotion_model.h5')
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

emotions1 = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

# -------- FEATURE EXTRACT --------
def zcr(data, frame_length, hop_length):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_feat = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feat.T) if not flatten else np.ravel(mfcc_feat.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    return np.hstack((
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.reshape(np.array(res), (1, 2376))
    i_result = scaler2.transform(result)
    return np.expand_dims(i_result, axis=2)

def predict_emotion(path):
    feat = get_predict_feat(path)
    predictions = emotion_model.predict(feat)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return emotions1.get(predicted_index + 1, "Unknown")

def preprocess_audio(file_path, sample_rate=16000, n_mfcc=20, max_frames=62):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]
    return torch.tensor(mfcc, dtype=torch.float32).T.unsqueeze(0).to(device)

def predict_gender(file_path):
    input_tensor = preprocess_audio(file_path)
    with torch.no_grad():
        output = gender_model(input_tensor)
        if output is None:
            print(f"⚠️ No output from gender model for: {file_path}")
            return "Unknown"
        prediction = torch.argmax(output, dim=1).item()
        return "Female" if prediction == 0 else "Male"


def predict_gender(file_path):
    input_tensor = preprocess_audio(file_path)
    with torch.no_grad():
        output = gender_model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return "Female" if prediction == 0 else "Male"

# -------- SPEAKER MATCHING --------
speaker_verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def get_matching_speaker(source_path, reference_folder, threshold=0.20):
    best_match = None
    best_score = -1
    for ref_file in glob.glob(os.path.join(reference_folder, "*.wav")):
        score, _ = speaker_verifier.verify_files(ref_file, source_path)
        st.write(f"🔎 Comparing with {os.path.basename(ref_file)} — Score: {score.item():.4f}")
        if score.item() > best_score:
            best_score = score.item()
            best_match = os.path.basename(ref_file)
    
    if best_score > threshold:
        return f"✅ Matched with {best_match} (Score: {best_score:.4f})"
    else:
        return f"❌ No match found (Highest Score: {best_score:.4f})"

# -------- LOAD SEPARATION MODELS --------
sep_model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')
enh_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham-enhancement')

def process_mixture(mix_path):
    est_sources = sep_model.separate_file(path=mix_path)
    os.makedirs("temp_sep", exist_ok=True)
    sep_paths = []
    for i in range(est_sources.shape[2]):
        sep_path = f"temp_sep/source_{i+1}.wav"
        torchaudio.save(sep_path, est_sources[:, :, i].detach().cpu(), 8000)
        sep_paths.append(sep_path)
    return sep_paths

def enhance_and_predict(paths, reference_folder="reference_speakers"):
    os.makedirs("enhanced_outputs", exist_ok=True)
    results = []
    for i, src_path in enumerate(paths):
        enhanced_sources = enh_model.separate_file(path=src_path)
        enhanced_path = f"enhanced_outputs/enhanced_source{i+1}.wav"
        torchaudio.save(enhanced_path, enhanced_sources[:, :, 0].detach().cpu(), 8000)

        gender = predict_gender(enhanced_path)
        emotion = predict_emotion(enhanced_path)
        match_result = get_matching_speaker(enhanced_path, reference_folder)

        results.append({
            "speaker": f"Speaker {i+1}",
            "gender": gender,
            "emotion": emotion,
            "match": match_result,
            "audio": enhanced_path
        })
    return results

# -------- STREAMLIT UI --------
st.sidebar.title("🎙️ Enhancing Forensic Audio Investigations:A Multi-Speaker Speech Understanding and Analysis System")
st.sidebar.markdown("""
### 📄 Project Description  
This project is a Speech Understanding System designed to analyze audio recordings containing multiple speakers.  
It performs the following:

- 🎙️ **Speaker Diarization** – Separate individual voices  
- 😊 **Emotion Classification** – Detect emotional tones  
- 🚻 **Gender Classification** – Predict male/female  
- 🧑‍💼 **Speaker Identification** – Match with reference samples

Upload a multi-speaker audio file from the main panel to see it in action!
""")

input_mix = st.sidebar.file_uploader("Upload Mixed Audio File (WAV)", type=["wav"])
ref_folder = st.sidebar.text_input("Path to Reference Speaker Folder", value="reference_speakers")

if input_mix is not None:
    with open("uploaded_mix.wav", "wb") as f:
        f.write(input_mix.read())

    st.subheader("Input Mixture Audio")
    st.audio("uploaded_mix.wav", format='audio/wav')
    st.write("🔄 Separating speakers...")

    sep_paths = process_mixture("uploaded_mix.wav")
    st.success(f"✅ {len(sep_paths)} speaker(s) separated.")

    st.write("🔧 Enhancing & analyzing each speaker...")
    results = enhance_and_predict(sep_paths, ref_folder)

    for res in results:
        st.subheader(res["speaker"])
        st.audio(res["audio"], format='audio/wav')
        st.write(f"**Gender:** {res['gender']}")
        st.write(f"**Emotion:** {res['emotion']}")
        st.write(f"**Speaker Match:** {res['match']}")
