# Music Gerne Classification using deep learning

The GTZAN dataset is a widely used music genre classification dataset in the field of machine learning and music information retrieval. It was created by George Tzanetakis and Perry Cook in 2002 and is named after their initials. The data set we have used has the song samples divided into ten genres naming: *disco, metal, reggae, blues, rock, classical, jazz, hip-hop, country and pop.*

Each audio track in the GTZAN dataset is sampled at 44.1kHz and is saved in uncompressed WAV format. The dataset also comes with metadata for each track, such as the artist, title, and genre label. The GTZAN dataset has been widely used to develop and evaluate music genre classification algorithms and has become a benchmark for comparing different methods.

# MFCC 
Mel Frequency Cepstral Coefficients (MFCCs) are a feature extraction technique widely used in audio signal processing, especially for speech and music recognition. MFCCs aim to capture the spectral characteristics of an audio signal by measuring the energy distribution in frequency bands that are spaced according to the Mel scale, which is a perceptual scale of pitches that is more closely related to the way humans hear sound.

The process of computing MFCCs involves several steps. First, the audio signal is divided into small frames, typically 20-30 milliseconds in duration. Next, a window function, such as the Hamming window, is applied to each frame to reduce spectral leakage. Then, the discrete Fourier transform (DFT) is applied to each frame to obtain its frequency spectrum. The resulting spectrum is then transformed using a filterbank that is designed to approximate the response of the human auditory system. The filterbank is typically logarithmic in frequency, with the filters spaced according to the Mel scale.

After the filterbank is applied, the logarithm of the filterbank energies is taken, and the resulting sequence of log filterbank energies is transformed using the Discrete Cosine Transform (DCT) to obtain a set of coefficients that capture the spectral characteristics of the signal. These coefficients are referred to as MFCCs.

MFCCs are widely used in applications such as speech recognition, speaker identification, and music genre classification. They are popular because they are effective at capturing the spectral characteristics of a signal while reducing the dimensionality of the data, which can make subsequent processing and analysis more efficient.


## Contributing

Contributions are welcome! If you find any issues or want to add new features, feel free to submit a pull request.

## Contact Me

<table>
  <tr>
    <td><img src="https://github.com/realsanjeev/protfolio/blob/main/src/assets/images/instagram.png" alt="Instagram" width="50" height="50"></td>
    <td><img src="https://github.com/realsanjeev/protfolio/blob/main/src/assets/images/twitter.png" alt="Twitter" width="50" height="50"></td>
    <td><img src="https://github.com/realsanjeev/protfolio/blob/main/src/assets/images/github.png" alt="GitHub" width="50" height="50"></td>
    <td><img src="https://github.com/realsanjeev/protfolio/blob/main/src/assets/images/linkedin-logo.png" alt="LinkedIn" width="50" height="50"></td>
  </tr>
</table>
