# Voice Biometrics System using GMM on TIMIT Dataset

This project is a command-line based speaker verification system built in Python. It uses Gaussian Mixture Models (GMMs) to create voice profiles for speakers and verify their identity. The system is designed to work with the TIMIT dataset.

The core logic involves extracting Mel-Frequency Cepstral Coefficients (MFCCs) from audio files, training a unique GMM for each speaker during an enrollment phase, and then scoring a test audio file against a claimed speaker's model for verification.

---

##  Features

- **Batch Enrollment**: Automatically enrolls all speakers from the TIMIT `TRAIN` directory upon startup.
- **Speaker Verification**: A simple CLI to verify a test audio file against an enrolled speaker's voice profile.
- **Configurable Parameters**: Easily change GMM and feature extraction settings in the `config.py` file.
- **Modular Code**: The project is broken down into logical modules for better readability and maintenance.

---

##  Acknowledgements
This project uses the TIMIT Acoustic-Phonetic Continuous Speech Corpus, developed by the Linguistic Data Consortium (LDC). All rights and credits for the dataset go to its original creators.

##  Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

- Python 3.7+
- The TIMIT Acoustic-Phonetic Continuous Speech Corpus. You need to have this dataset downloaded on your computer.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-link>
    cd Voice-Biometrics-GMM
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
    ```

4.  ** Configure the Dataset Path:**
    Open the `config.py` file and **change the `TIMIT_DATASET_FOLDER_ON_DISK` variable** to the absolute path of your TIMIT dataset folder.
    ```python
    # in config.py
    TIMIT_DATASET_FOLDER_ON_DISK = '/path/to/your/timit/data' # <-- CHANGE THIS
    ```

---

##  How to Run

Once the setup is complete, you can run the main application from your terminal:

```sh
python main.py
