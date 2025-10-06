import os
import glob
from gmm_handler import verify_speaker

def list_speakers(speaker_dict, set_name):
    print(f"\n--- Available Speakers ({set_name} set) ---")
    if not speaker_dict:
        print(f"No speakers loaded from the {set_name} set.")
        return []
    
    speakers = sorted(speaker_dict.keys())
    print(f"Found {len(speakers)} speakers in the {set_name} set:")
    for spk in speakers:
        print(f"- {spk} ({len(speaker_dict[spk])} files)")
    print("---------------------------------------")
    return speakers

def list_enrolled_models(config):
    print(f"\n--- Enrolled Speaker Models (in {config.MODEL_DIR}) ---")
    try:
        model_files = glob.glob(os.path.join(config.MODEL_DIR, '*.gmm'))
        if not model_files:
            print("No enrolled models (.gmm files) found yet.")
            return []
        
        enrolled = [os.path.basename(mf).replace('.gmm', '') for mf in model_files]
        print(f"Found {len(enrolled)} enrolled models:")
        for speaker_id in sorted(enrolled):
            print(f"- {speaker_id}")
        print("---------------------------------------------")
        return sorted(enrolled)
    except Exception as e:
        print(f"Error listing model files: {e}")
        return []

def cli_verify(enroll_files, test_files, config):
    print("\n===== Speaker Verification Menu =====")
    
    enrolled_speakers = list_enrolled_models(config)
    if not enrolled_speakers:
        print("Cannot verify: No speakers have been enrolled yet.")
        return

    claimed_speaker_id = input(f"Enter the claimed Speaker ID (must be enrolled): ").strip()
    if claimed_speaker_id not in enrolled_speakers:
        print(f"Error: Speaker '{claimed_speaker_id}' is not enrolled.")
        return

    print("\nSelect a speaker from the TEST set to use their audio file for verification:")
    test_speaker_list = list_speakers(test_files, "TEST")
    if not test_speaker_list:
        return
    test_speaker_id = input("Enter Speaker ID from TEST set: ").strip()
    if test_speaker_id not in test_files:
        print(f"Error: Speaker ID '{test_speaker_id}' not found in TEST set.")
        return

    test_audio_file = test_files[test_speaker_id][0]
    print(f"\nUsing test file: {os.path.basename(test_audio_file)}")
    print(f"(True speaker: {test_speaker_id})")

    print(f"\nVerifying against model for '{claimed_speaker_id}'...")
    score = verify_speaker(claimed_speaker_id, test_audio_file, config)

    if score is not None:
        print(f"\nVerification Score (Avg Log-Likelihood): {score:.4f}")
        print(f"Decision Threshold: {config.VERIFICATION_THRESHOLD}")
        
        is_accepted = score > config.VERIFICATION_THRESHOLD
        is_genuine = (claimed_speaker_id == test_speaker_id)
        
        result = "ACCEPTED" if is_accepted else "REJECTED"
        print(f"Result: {result}")
        
        if is_genuine and is_accepted:
            print("  -> Correct Acceptance")
        elif not is_genuine and not is_accepted:
            print("  -> Correct Rejection")
        elif is_genuine and not is_accepted:
            print("  -> False Rejection!")
        elif not is_genuine and is_accepted:
            print("  -> False Acceptance!")
    else:
        print("\nResult: Verification FAILED .")
    print("===================================")