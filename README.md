# Audio Comparison App

This Python application provides functionality to compare two audio files and provide a similarity score. It uses various audio features, such as chroma features, tempo, and pitch, to compute the similarity between the reference and query audio.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3
- Required Python packages (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd audio-comparison-app
    ```
2. Install Dependancies

   ```bash
   pip install -r requirements.txt
   ```
### Usage
1. Run the Flask application:
   
   ```bash
   python app.py
   ```
The application will be accessible at http://localhost:5000.

2. Send a POST request to the /match/CompareAudio endpoint with the reference and query audio files.

Example using curl:
   
   ```bash
   curl -X POST -F "ref_audio_name=reference_audio" -F "q_audio=@/path/to/query_audio.wav" -F "uid=123" http://localhost:5000/match/CompareAudio
   ```
Replace reference_audio with the name of the reference audio file and /path/to/query_audio.wav with the path to the query audio file.

3. The response will include a label and a similarity score based on the comparison.

### Features
- Chroma Features: The application uses chroma features computed from the short-term Fourier transform (STFT), constant-Q transform (CQT), and other methods.
- Similarity Measures: The similarity between the reference and query audio is computed using cross-recurrent plots and various similarity measures like Qmax, Dmax, and more.
- Scoring: The application provides scores for consistency, pronunciation, intonation, pitch, and speed.

### Database Integration
The application integrates with a MySQL database to store and update user scores. Make sure to set up the database connection parameters in the .env file.
