import os
import math 
import librosa
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

SAMPLE_RATE = 22050
SEGMENT_LENGTH = 6 #in sec
MODEL_PATH = "./models/my_cnn_model_for_mfcc.h5"
GENRES = ['blues', 'hiphop', 'rock', 'pop', 'disco', 'reggae', 'jazz', 'country', 'classical', 'metal']
model = load_model("./models/my_cnn_model_for_mfcc.h5")

def get_mfcc(music_path, parts=5, n_mfcc=13, n_fft=2048, hop_length=512):
  # Divide given music in  6 sec segment
  samples_per_segment = SEGMENT_LENGTH* 22050
  mfcc_per_segment = math.ceil(samples_per_segment / hop_length)
  try:
    y, sr = librosa.load(music_path, sr=SAMPLE_RATE)
    durations = librosa.get_duration(y=y, sr=sr)
    parts = int(durations/ SEGMENT_LENGTH)
  except:
    print(f"[INFO] Please check given music path: {music_path}.")
    return None
  music_all_mfcc = []
  for part in range(parts):
  # find the start and finish of segment
    start = samples_per_segment * part
    end = start + samples_per_segment

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=y[start:end], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T

    if len(mfcc) == mfcc_per_segment:
      # resize to fit in cnn model
      mfcc = np.resize(mfcc, (mfcc.shape[0], mfcc.shape[1], 1))
      music_all_mfcc.append(mfcc)
    else:
      print(f"[INFO] {part} segment of {file} is not included ")
  return np.array(music_all_mfcc)
  

def prediction(test_mfcc):
  prediction = model.predict(test_mfcc)
  return prediction

def probability_graph_path(prediction):
  mean_pred = np.mean(prediction, axis=0)
  plt.figure(figsize=(10, 5))
  plt.bar(GENRES, height=mean_pred*100)

  # Plot for percentage of confidence
  plt.title("Prediction Confidence")
  plt.xlabel("Labels")
  plt.ylabel("Perrcentage of confidence")
  plt.ylim(0,100)
  path = "probability_graph.png"
  if os.path.exists(f"static/{path}"):
    os.remove(f"static/{path}")
    print(f"File 'static/{path}' deleted successfully.")
  else:
    print(f"File 'static/{path}' does not exist.")
  plt.savefig(f"static/{path}", facecolor='y', bbox_inches="tight",
            pad_inches=0.3, transparent=True)
  
  print(F"[INFO] LABEL: {np.argmax(mean_pred)}")
  genre_result = GENRES[np.argmax(mean_pred)]

  return path, genre_result