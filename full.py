# !!! UWAGA !!!
# to jest wersja dla analizy CAŁOŚCI NAGRANIA
# zatem nie będzie można jej przenieść bezpośrednio na działanie w czasie rzeczywistym
# może posłużyć jako utworzenie danych referencyjnych do porównania dla innych metod

# TODO:
# 1. wczytanie danych z pliku wav
# 2. obliczenie maksymalnego możliwego opóźnienia w próbkach
# 3. wykorznanie gcc-phat dla każdej pary mikrofonów
# 4. analiza trajektori algorytmem Viterbiego
# 5. obliczenie trajektorii na płaszczyźnie
# 6. wizualizacja trajektorii na płaszczyźnie z zaznaczonymi krzywymi (hiperbolami)
# 7. zapisanie wyników w jakiś uporządkowany sposób
# 8. wyświetlanie czasu po każdym etapie

import time
import numpy as np
import scipy as sp
import acoular as ac
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

wavFiles = [
    './audio/TwoDrones_96000Hz_24bit.wav',
    './audio/20250612_1545_8ch_aligned.wav',
    './audio/car_aligned.wav',
    './audio/two_drones_aligned.wav'
]

n = 8192

# True aby wyświetlać wykresy na bieżąco
testPlots = True
# True aby zapisywać dane
saveData = False
# prefiks dodawany plików przy zapisie
filePrefix = ''

def loadMicrophonePositions(xml_file):
    mg = ac.MicGeom(file = xml_file)
    return mg.pos[:2, :].T

def calculateDelayRange(sampleRate, c = 343.0):
    # maksymalne możliwe opoźnienie w próbkac
    microphoneDistance = 0.23 # [m]
    delayRange = microphoneDistance * sampleRate / c
    return int(delayRange // 2 + 1) + 10

def timeSamplesGeneratorFromWavFile(filename, n = 8192):
    sampleRate, wavData = sp.io.wavfile.read(filename)

    numberOfSamples = wavData.shape[0]
    numberOfChannels = wavData.shape[1] if wavData.ndim > 1 else 1
    duration = numberOfSamples / sampleRate

    print("")
    print("=" * 50)
    print(f"File:       {filename}")
    print(f"Samples:    {numberOfSamples}")
    print(f"Channels:   {numberOfChannels}")
    print(f"SampleRate: {sampleRate} Hz")
    print(f"Duration:   {duration} s")
    print(f"Frames:     {int(np.ceil(numberOfSamples / n))}")
    print("=" * 50, end='\n\n')

    timeSamples = ac.TimeSamples(data = wavData, sample_freq = sampleRate)
    return timeSamples.result(num = n), numberOfChannels, sampleRate

def gccPhat(xseg, yseg, n = 8192, delayRange = 50):
    minDelay = n//2 - delayRange
    maxDelay = n//2 + delayRange

    xseg = xseg * np.hanning(len(xseg))
    yseg = yseg * np.hanning(len(yseg))

    X = np.fft.fft(xseg, n = n)
    Y = np.fft.fft(yseg, n = n)

    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-12
    cc = np.fft.ifft(R, n = n)
    cc = np.fft.fftshift(cc).real

    return cc[minDelay:maxDelay]

def viterbiAlgorithm(transitionMatrix, emissionMatrix, initialProbability, emissionWeight = 1.9):
    A = np.array(transitionMatrix)
    B = np.array(emissionMatrix)
    pi = np.array(initialProbability)

    eps = 1e-12

    log_A = np.log(A + eps)
    log_B = np.log(B + eps) * emissionWeight
    log_pi = np.log(pi + eps)

    y_dim, x_dim = B.shape

    V = np.zeros((y_dim, x_dim))
    V_from = np.zeros((y_dim, x_dim), dtype=int)

    V[:, 0] = log_pi + log_B[:, 0]

    for t in range(1, x_dim):
        transitionProb = V[:, t - 1, None] + log_A
        V[:, t] = np.max(transitionProb, axis = 0) + log_B[:, t]
        V_from[:, t] = np.argmax(transitionProb, axis = 0)

    states = np.zeros(x_dim, dtype = int)
    states[-1] = np.argmax(V[:, -1])

    for t in range(x_dim - 2, -1, -1):
        states[t] = V_from[states[t+1], t+1]

    return states

def gaussianDistributionFromPeaks(peaks, length, sigma = 3.0, normalize = True):
    y = np.arange(length)
    prior = np.zeros(length, dtype=float)

    for p in peaks:
        prior += np.exp(-(y - p)*(y - p) / (2 * sigma * sigma))

    if normalize:
        prior /= prior.sum() + 1e-12

    return prior

def removePathFromHeatmap(heatmap, path, radius = 6):
    new_heatmap = heatmap.copy()
    y_dim, x_dim = new_heatmap.shape
    for x, y_coord in enumerate(path):
        y_min = max(0, y_coord - radius)
        y_max = min(y_dim, y_coord + radius + 1)
        new_heatmap[y_min:y_max, x] = 0.0
    return new_heatmap

def calculatePathsVA(heatmap, numberOfLines = 2, sigma = 1.0, emissionWeight = 5.0, removeRadius = 3):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
    yRange, xRange = heatmap.shape

    transitionMatrix = np.zeros((yRange, yRange))
    for i in range(yRange):
        for j in range(yRange):
            transitionMatrix[i, j] = np.exp(-(i - j) * (i - j) / (2 * sigma * sigma))
    # prawa macierz stochastyczna (wiersz)
    transitionMatrix /= transitionMatrix.sum(axis = 1, keepdims = True)
    currentHeatmap = heatmap.copy()
    allPaths = []

    for i in range(numberOfLines):
        peak = np.argmax(heatmap[:, 0])

        initialProbability = gaussianDistributionFromPeaks(
            peaks = [peak],
            length = yRange,
            sigma = 10.0,
            normalize = True
        )

        path = viterbiAlgorithm(
            transitionMatrix = transitionMatrix,
            emissionMatrix = currentHeatmap,
            initialProbability = initialProbability,
            emissionWeight = emissionWeight
        )
        allPaths.append(path)

        currentHeatmap = removePathFromHeatmap(
            heatmap = currentHeatmap,
            path = path,
            radius = removeRadius
        )

    return np.array(allPaths)

def hyperbolaContours(allPaths, micPairs, micPositions, X, Y, c = 343.0):
    # allPaths.shape -> (4, 2, 1259)
    # zawiera 2 polozenia maksimow dla każdego z 4 mikrofonów w każdej z 1259 ramek

    # dla kżdej ramki pozycja (x,y) dla źródła (1259, 2, 2)
    est = np.zeros((allPaths.shape[2], allPaths.shape[1], 2))

    # łatwiej będie analizować dla kojedyńczych ramek
    for frame_idx in range(allPaths.shape[2]):
        print(f"\r{frame_idx + 1}", end = '')

        errorMap = np.zeros((allPaths.shape[1], *X.shape))

        # teraz po każdej parze mikrofonow
        for mic_idx in range(allPaths.shape[0]):
            mic1_idx = micPairs[mic_idx][0] # numer 1 mikrofonu
            mic2_idx = micPairs[mic_idx][1] # numer 2 mikrofonu
            mic1_pos = micPositions[mic1_idx]
            mic2_pos = micPositions[mic2_idx]

            d1x = X - mic1_pos[0]
            d1y = Y - mic1_pos[1]
            d2x = X - mic2_pos[0]
            d2y = Y - mic2_pos[1]

            d1 = np.sqrt(d1x * d1x + d1y * d1y)
            d2 = np.sqrt(d2x * d2x + d2y * d2y)

            H = d1 - d2

            # dla każdego potencjalnego źródła
            for src_idx in range(allPaths.shape[1]):
                tdoa = allPaths[mic_idx, src_idx, frame_idx]
                delta_d = c * tdoa
                H1 = H - delta_d
                errorMap[src_idx] += H1 * H1

        errorMap = np.array(errorMap)

        for i in range(errorMap.shape[0]):
            min_idx = np.unravel_index(np.argmin(errorMap[i]), errorMap[i].shape)
            est_x = X[min_idx]
            est_y = Y[min_idx]
            est[frame_idx, i] = [est_x, est_y]

    print("")
    return est

def animateTrajectories(pos, interval = 20, tail_length = 50):
    n_frames = pos.shape[0]
    n_sources = pos.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))

    margin = 0.02
    ax.set_xlim(np.min(pos[:, :, 0]) - margin, np.max(pos[:, :, 0]) + margin)
    ax.set_ylim(np.min(pos[:, :, 1]) - margin, np.max(pos[:, :, 1]) + margin)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Animacja")

    lines = []
    points = []
    colors = ['#1f77b4', '#ff7f0e']

    for i in range(n_sources):
        line, = ax.plot([], [], color=colors[i], alpha=0.5, label=f'Trajektoria {i + 1}')
        point, = ax.plot([], [], color=colors[i], marker='o', markersize=8, markeredgecolor='black')
        lines.append(line)
        points.append(point)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend()

    def update(frame):
        for i in range(n_sources):
            start_idx = 0 # if tail_length is None else max(0, frame - tail_length)

            x_history = pos[start_idx:frame + 1, i, 0]
            y_history = pos[start_idx:frame + 1, i, 1]

            lines[i].set_data(x_history, y_history)
            points[i].set_data([pos[frame, i, 0]], [pos[frame, i, 1]])

        time_text.set_text(f'Klatka: {frame + 1}')
        return lines + points + [time_text]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()
    return ani

def main():
    t_start = time.thread_time()

    # DATA
    timeSamplesGenerator, numberOfChannels, sampleRate = timeSamplesGeneratorFromWavFile(wavFiles[0], n = n)
    delayRange = calculateDelayRange(sampleRate)
    print(f"DelayRange: {delayRange}")

    t_data = time.thread_time()
    print(f"DATA:       {t_data - t_start} s")

    # GCC-PHAT
    ccs = [[] for _ in range(numberOfChannels // 2)]

    for frame, block in enumerate(timeSamplesGenerator):
        print(f"\r{frame + 1}", end = "")

        for i in range(numberOfChannels//2):
            channel1 = i
            channel2 = i + numberOfChannels//2

            xseg = block[:, channel1]
            yseg = block[:, channel2]

            cc = gccPhat(
                xseg = xseg,
                yseg = yseg,
                n = n,
                delayRange = delayRange
            )
            ccs[i].append(cc)

    print("")
    ccs = np.array(ccs)

    if saveData:
        np.save('./data/ccs_96.npy', ccs)

    # TEST
    if testPlots:
        for heatmap in ccs:
            plt.figure(figsize=(8, 5))
            plt.imshow(heatmap.T, cmap='gray', aspect='auto', origin='lower')
            plt.tight_layout()
            plt.show()

    t_gcc = time.thread_time()
    print(f"GCC-PHAT:   {t_gcc - t_data} s")

    # VITERBI

    numberOfLines = 2
    sigma = 1.0
    emissionWeight = 5.0
    removeRadius = 3

    allPaths = []

    for i in range(ccs.shape[0]):
        heatmap = ccs[i]
        heatmap = heatmap.T

        paths = calculatePathsVA(
            heatmap = heatmap,
            numberOfLines = numberOfLines,
            sigma = sigma,
            emissionWeight = emissionWeight,
            removeRadius = removeRadius
        )

        allPaths.append(paths)

        # TEST
        if testPlots:
            plt.figure(figsize=(8, 5))
            plt.imshow(heatmap, cmap='gray', aspect='auto', origin='lower')
            for p in paths:
                plt.plot(p, linewidth=2)
            plt.tight_layout()
            plt.show()

    allPaths = np.array(allPaths)

    if saveData:
        np.save('data/allPaths_tda.npy', allPaths)

    t_viterbi = time.thread_time()
    print(f"VITERBI:    {t_viterbi - t_gcc} s")

    # HYPERBOLE
    microphonePositions = loadMicrophonePositions("./xml/ring8_capstone_wall.xml")
    microphonePairs = [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    x = np.linspace(-0.2, 0.2, 600)
    y = np.linspace(-0.2, 0.2, 600)
    X, Y = np.meshgrid(x, y)

    # zamiana na sekundy
    allPaths = (allPaths - delayRange) / sampleRate

    pos = hyperbolaContours(
        allPaths = allPaths,
        micPairs = microphonePairs,
        micPositions = microphonePositions,
        X = X,
        Y = Y
    )

    # np.save('data/trajectories.npy', allPaths)

    print(f"HYPERBOLE:  {time.thread_time() - t_viterbi} s")

    # ANIMATION
    ani = animateTrajectories(pos)

    if saveData:
        print('Saving animation...')
        ani.save('./hiperbole/trajektoria.mp4', writer='ffmpeg', fps=30)

    t_end = time.thread_time()

    return t_end - t_start

if __name__ == '__main__':
    time = main()
    print(f'TIME:       {time} s')
