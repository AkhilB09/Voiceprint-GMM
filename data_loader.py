import os
import glob
from collections import defaultdict

def find_timit_files(timit_base_path):
    print(f"\n--- Attempting to load TIMIT .WAV file paths from: {timit_base_path} ---")

    if not os.path.exists(timit_base_path):
        print(f"Error: TIMIT dataset path '{timit_base_path}' does not exist. Please check config.py.")
        return None, None

    enroll_files = defaultdict(list)
    test_files = defaultdict(list)

    print("\nProcessing TRAIN directory...")
    train_files_count, train_speakers = _process_subset(timit_base_path, 'TRAIN', enroll_files)
    print(f"Found {train_files_count} files for {len(train_speakers)} speakers (for enrollment).")

    print("\nProcessing TEST directory...")
    test_files_count, test_speakers = _process_subset(timit_base_path, 'TEST', test_files)
    print(f"Found {test_files_count} files for {len(test_speakers)} speakers (for verification).")

    if not enroll_files and not test_files:
        print("\nWarning: No speaker files were loaded. Please check the directory structure.")

    return enroll_files, test_files

def _process_subset(base_path, subset_dir, file_dict):
    files_found = 0
    speakers_found = set()
    subset_path = os.path.join(base_path, subset_dir)

    if not os.path.exists(subset_path):
        print(f"Warning: Directory not found: {subset_path}")
        return 0, set()

    dialect_regions = glob.glob(os.path.join(subset_path, 'DR*'))

    if not dialect_regions:
        print(f"Warning: No DR* folders found in {subset_path}. Looking for speaker folders directly...")
        speaker_paths = glob.glob(os.path.join(subset_path, '*'))
        if any(os.path.isdir(p) for p in speaker_paths):
            dialect_regions = [subset_path]
        else:
            print(f"Warning: No speaker folders found here either.")
            return 0, set()

    for dr_path in dialect_regions:
        speaker_paths = glob.glob(os.path.join(dr_path, '*'))
        for spk_path in speaker_paths:
            if not os.path.isdir(spk_path):
                continue

            speaker_id = os.path.basename(spk_path)
            wav_files = glob.glob(os.path.join(spk_path, '*.WAV')) + \
                        glob.glob(os.path.join(spk_path, '*.wav'))

            if wav_files:
                file_dict[speaker_id].extend(wav_files)
                files_found += len(wav_files)
                speakers_found.add(speaker_id)

    return files_found, speakers_found