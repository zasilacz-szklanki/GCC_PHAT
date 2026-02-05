import scipy as sp

def wav_info(filename):
    sampleRate, wavData = sp.io.wavfile.read(filename)

    n_samples = wavData.shape[0]
    n_channels = wavData.shape[1] if wavData.ndim > 1 else 1
    duration = n_samples / sampleRate

    print("=" * 40)
    print(f"Plik: {filename}")
    print(f"Liczba próbek: {n_samples}")
    print(f"Liczba kanałów: {n_channels}")
    print(f"Częstotliwość próbkowania: {sampleRate} Hz")
    print(f"Czas trwania: {duration} s")
    print("=" * 40, end='\n\n')

audio = [
    './audio/TwoDrones_96000Hz_24bit.wav',
    './audio/20250612_1545_8ch_aligned.wav',
    './audio/car_aligned.wav',
    './audio/two_drones_aligned.wav'
]

for f in audio:
    wav_info(f)
