import os
import wave
import numpy as np
import pyaudio
import speech_recognition as sr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Function to record audio
def record_audio(filename, duration=10, sample_rate=44100):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Function to extract features from audio for speaker differentiation
def extract_features(filename):
    with wave.open(filename, 'r') as wf:
        frames = wf.readframes(-1)
        waveform = np.frombuffer(frames, dtype=np.int16)

    # Compute simple features like mean, variance, and zero-crossing rate
    mean = np.mean(waveform)
    variance = np.var(waveform)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(waveform))))

    return [mean, variance, zero_crossings]

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Error with recognition service"

# Main program
def main():
    num_speakers = int(input("Enter the number of speakers to differentiate: "))
    filenames = []

    for i in range(num_speakers):
        filename = f"speaker_{i + 1}.wav"
        filenames.append(filename)
        print(f"Recording for Speaker {i + 1}...")
        record_audio(filename, duration=10)

    print("Extracting features and clustering...")
    features = np.array([extract_features(f) for f in filenames])

    # Using KMeans for clustering based on features
    clustering_model = KMeans(n_clusters=num_speakers)
    labels = clustering_model.fit_predict(features)

    print("Assigning speakers and transcribing:")
    for i, filename in enumerate(filenames):
        transcription = transcribe_audio(filename)
        speaker_label = labels[i]
        print(f"Speaker {speaker_label + 1}: {transcription}")

if __name__ == "__main__":
    main()
