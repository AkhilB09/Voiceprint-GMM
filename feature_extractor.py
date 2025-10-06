import os
import numpy as np
import librosa
import soundfile as sf
import warnings

def extract_mfcc(audio_path, config):
    if not os.path.exists(audio_path):
        return None
    try:
        y, sr_orig = sf.read(audio_path)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr_orig != config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=config.SAMPLE_RATE)

        if len(y) < int(0.1 * config.SAMPLE_RATE):
            return None

        n_fft = int(config.FRAME_LENGTH_MS / 1000 * config.SAMPLE_RATE)
        hop_length = int(config.HOP_LENGTH_MS / 1000 * config.SAMPLE_RATE)

        mfccs = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
                                     n_fft=n_fft, hop_length=hop_length)

        if config.INCLUDE_DELTAS:
            delta1 = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            features = np.concatenate((mfccs, delta1, delta2), axis=0)
        else:
            features = mfccs

        if features.shape[1] > 1:
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
            std[std < 1e-8] = 1e-8
            features = (features - mean) / std
        elif features.shape[1] == 1:
            features = features - np.mean(features, axis=1, keepdims=True)
        else:
            return None

        return features.T

    except Exception as e:
        print(f"Warning: Error processing '{os.path.basename(audio_path)}': {e}")
        return None