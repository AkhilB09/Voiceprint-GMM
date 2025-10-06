import os
import time
import warnings

import config
import data_loader
import gmm_handler
import cli_interface

warnings.filterwarnings('ignore', category=RuntimeWarning)

def run_batch_enrollment(enroll_files, config):
    print("\n--- Starting Batch Enrollment ---")
    start_time = time.time()
    
    total_speakers = len(enroll_files)
    enrolled_count = 0
    failed_speakers = []

    print(f"Attempting to enroll {total_speakers} speakers...")

    for i, (speaker_id, files) in enumerate(enroll_files.items()):
        print(f"Enrolling... ({i+1}/{total_speakers}): {speaker_id}")
        success = gmm_handler.enroll_speaker(speaker_id, files, config)
        if success:
            enrolled_count += 1
        else:
            failed_speakers.append(speaker_id)
            print(f"  --> FAILED to enroll {speaker_id}.")
    
    end_time = time.time()
    print("\n--- Batch Enrollment Summary ---")
    print(f"Successfully enrolled {enrolled_count} out of {total_speakers} speakers.")
    if failed_speakers:
        print(f"Failed speakers: {', '.join(failed_speakers)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print("---------------------------------")
    return enrolled_count


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.FEATURE_DIR, exist_ok=True)

    enroll_files, test_files = data_loader.find_timit_files(config.TIMIT_DATASET_FOLDER_ON_DISK)
    
    if enroll_files is None or not enroll_files:
        print("\nNo files found for enrollment. Exiting script.")
        return

    enrolled_count = run_batch_enrollment(enroll_files, config)
    
    if enrolled_count == 0:
        print("\nNo speakers were successfully enrolled. Cannot start the CLI.")
        return

    while True:
        print("\n--- Main Menu ---")
        print("1: Verify Speaker")
        print("2: List Available Speakers (TRAIN set)")
        print("3: List Available Speakers (TEST set)")
        print("4: List Enrolled Models")
        print("5: Exit")
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            cli_interface.cli_verify(enroll_files, test_files, config)
        elif choice == '2':
            cli_interface.list_speakers(enroll_files, "TRAIN")
        elif choice == '3':
            cli_interface.list_speakers(test_files, "TEST")
        elif choice == '4':
            cli_interface.list_enrolled_models(config)
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select from 1-5.")

if __name__ == '__main__':
    print("===========================================")
    print("==== Voice BioMetrics( GMM) System Start ===")
    print("===========================================")
    main()
    print("\n--- Script Execution Finished ---")