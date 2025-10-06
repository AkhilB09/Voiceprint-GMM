import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from feature_extractor import extract_mfcc

def enroll_speaker(speaker_id, audio_files, config):
    model_path = os.path.join(config.MODEL_DIR, f"{speaker_id}.gmm")
    all_enroll_features = []

    for audio_file in audio_files:
        features = extract_mfcc(audio_file, config)
        if features is not None and features.shape[0] > 0:
            all_enroll_features.append(features)

    if not all_enroll_features:
        print(f"Error: No valid features extracted for {speaker_id}.")
        return False

    try:
        enroll_features_matrix = np.vstack(all_enroll_features)
    except ValueError as e:
        print(f"Error stacking features for {speaker_id}: {e}")
        return False

    n_frames, _ = enroll_features_matrix.shape

    if n_frames < config.N_COMPONENTS:
        print(f"Error: Fewer frames ({n_frames}) than GMM components ({config.N_COMPONENTS}) for {speaker_id}. Cannot enroll.")
        return False

    try:
        gmm = GaussianMixture(n_components=config.N_COMPONENTS,
                              covariance_type=config.COVARIANCE_TYPE,
                              reg_covar=config.REG_COVAR,
                              max_iter=config.MAX_ITER_GMM,
                              random_state=0,
                              verbose=0)
        gmm.fit(enroll_features_matrix)

        with open(model_path, 'wb') as f_model:
            pickle.dump(gmm, f_model)

        return True

    except Exception as e:
        print(f"Error during GMM training for '{speaker_id}': {e}")
        return False

def verify_speaker(claimed_speaker_id, test_audio_file, config):
    model_path = os.path.join(config.MODEL_DIR, f"{claimed_speaker_id}.gmm")

    if not os.path.exists(model_path):
        print(f"Error: Model not found for speaker '{claimed_speaker_id}'.")
        return None

    test_features = extract_mfcc(test_audio_file, config)
    if test_features is None or test_features.shape[0] == 0:
        print(f"Error: Could not extract features from test file: {os.path.basename(test_audio_file)}.")
        return None

    try:
        with open(model_path, 'rb') as f_model:
            gmm = pickle.load(f_model)
        
        log_likelihood_score = gmm.score(test_features)
        return log_likelihood_score

    except Exception as e:
        print(f"Error during verification scoring: {e}")
        return None