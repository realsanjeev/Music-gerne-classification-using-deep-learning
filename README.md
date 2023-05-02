# Music_gerne_classification using deep learning
This project havw used the GTZAN dataset which consists of 1000 songs of different genres in a managed order. The data set we have used has the song samples divided into ten genres naming: *disco, metal, reggae, blues, rock, classical, jazz, hip-hop, country and pop.*

# MFCC 
Mel Frequency Cepstral Coefficients (MFCCs) are a feature extraction technique widely used in audio signal processing, especially for speech and music recognition. MFCCs aim to capture the spectral characteristics of an audio signal by measuring the energy distribution in frequency bands that are spaced according to the Mel scale, which is a perceptual scale of pitches that is more closely related to the way humans hear sound.

The process of computing MFCCs involves several steps. First, the audio signal is divided into small frames, typically 20-30 milliseconds in duration. Next, a window function, such as the Hamming window, is applied to each frame to reduce spectral leakage. Then, the discrete Fourier transform (DFT) is applied to each frame to obtain its frequency spectrum. The resulting spectrum is then transformed using a filterbank that is designed to approximate the response of the human auditory system. The filterbank is typically logarithmic in frequency, with the filters spaced according to the Mel scale.

After the filterbank is applied, the logarithm of the filterbank energies is taken, and the resulting sequence of log filterbank energies is transformed using the Discrete Cosine Transform (DCT) to obtain a set of coefficients that capture the spectral characteristics of the signal. These coefficients are referred to as MFCCs.

MFCCs are widely used in applications such as speech recognition, speaker identification, and music genre classification. They are popular because they are effective at capturing the spectral characteristics of a signal while reducing the dimensionality of the data, which can make subsequent processing and analysis more efficient.



