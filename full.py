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

# wyświetlanie czasu po każdym etapie

import time
import numpy as np
import scipy as sp
import acoular as ac
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TO REMOVE
def plotHeatmap2(matrix, cc_maxs=None, name='gp5', min_max=None, zero=None, yticks=None, yticks_step = 5, ylim=None, ch1 = -1, ch2 = -1):

    plt.figure(figsize=(8, 4))
    #
    if ylim is not None:
        plt.ylim(ylim)
    #
    # # --- USTALENIE EXTENT ---
    # if yticks is not None:
    #     y_min = np.min(yticks)
    #     y_max = np.max(yticks)
    # else:
    #     y_min = 0
    #     y_max = matrix.shape[1]
    #
    # x_min = 0
    # x_max = matrix.shape[0]

    # --- HEATMAPA Z EXTENT ---
    plt.imshow(
        matrix.T,
        aspect='auto',
        cmap='gray',
        origin='lower',
        # extent=(0, len(matrix), min_delay - n // 2, max_delay - n // 2),
    )
    # plt.colorbar(label='Wartość')

    # # --- OŚ X ---
    x = np.arange(matrix.shape[0])
    #
    # # --- OŚ Y ---
    # if yticks is not None:
    #     ymin = int(np.ceil(yticks[0]))
    #     ymax = int(np.floor(yticks[-1]))
    #     # plt.yticks(np.arange(ymin, ymax + 1, yticks_step))
    #
    # # --- ZERO LINE ---
    if zero is not None:
        plt.plot(x, np.zeros_like(x) + zero, linewidth=2, color='yellow', linestyle='dashed')
    #
    # # --- CC MAXS ---
    if cc_maxs is not None:
        plt.plot(x, np.array(cc_maxs), 'r', linewidth=2)
    #
    # # --- MIN/MAX ---
    # if min_max is not None:
    #     plt.plot(x, np.zeros_like(x) + min_max[0], 'w', linewidth=2)
    #     plt.plot(x, np.zeros_like(x) + min_max[1], 'w', linewidth=2)

    plt.xlabel("Numer ramki w nagraniu")
    plt.ylabel("Przesunięcie w próbkach")
    plt.title(f"Funkcja korelacji {matrix.T.shape} Kanały: {ch1}, {ch2}")

    # plt.tight_layout()
    plt.show()

wavFiles = [
    './audio/TwoDrones_96000Hz_24bit.wav',
    './audio/20250612_1545_8ch_aligned.wav',
    './audio/car_aligned.wav',
    './audio/two_drones_aligned.wav'
]

n = 8192
delayRange = 50

def loadMicrophonePositions(xml_file):
    mg = ac.MicGeom(file = xml_file)
    xs = mg.pos[0]
    ys = mg.pos[1]
    return list(zip(xs, ys))

def timeSamplesGeneratorFromWavFile(filename, n = 8192):
    sampleRate, wavData = sp.io.wavfile.read(filename)
    numberOfChannels = wavData.shape[1] if wavData.ndim > 1 else 1
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

def viterbiAlgorithm(transitionMatrix, emissionMatrix, initialProbability, emissionWeight=1.9):
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

def gaussianDistributionFromPeaks(peaks, length, sigma=3.0, normalize=True):
    y = np.arange(length)
    prior = np.zeros(length, dtype=float)

    for p in peaks:
        prior += np.exp(-(y - p)**2 / (2 * sigma**2))

    if normalize:
        prior /= prior.sum() + 1e-12

    return prior

def removePathFromHeatmap(heatmap, path, radius=6):
    new_heatmap = heatmap.copy()
    y_dim, x_dim = new_heatmap.shape
    for x, y_coord in enumerate(path):
        y_min = max(0, y_coord - radius)
        y_max = min(y_dim, y_coord + radius + 1)
        new_heatmap[y_min:y_max, x] = 0.0
    return new_heatmap

def calculatePathsVA(heatmap, numberOfLines = 2, beta = 2.0, emissionWeight = 1.9, removeRadius = 3):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
    yRange, xRange = heatmap.shape

    transitionMatrix = np.zeros((yRange, yRange))
    for i in range(yRange):
        for j in range(yRange):
            transitionMatrix[i, j] = np.exp(-(i - j)*(i - j) / (beta * beta))
    transitionMatrix /= transitionMatrix.sum(axis=1, keepdims=True)
    currentHeatmap = heatmap.copy()
    allPaths = []

    for i in range(numberOfLines):
        peak = np.argmax(heatmap[:, 0])

        initialProbability = gaussianDistributionFromPeaks(
            peaks=[peak],
            length=yRange,
            sigma=10.0,
            normalize=True
        )

        path = viterbiAlgorithm(transitionMatrix, currentHeatmap, initialProbability, emissionWeight)
        allPaths.append(path)

        currentHeatmap = removePathFromHeatmap(currentHeatmap, path, radius=removeRadius)

    return np.array(allPaths)

def hyperbolaContours(tdoa_list, mic_pairs, mic_positions, X, Y, c = 343.0):
    # tdoa_list.shape -> (4, 2, 1259)
    # zawiera 2 polozenia maksimow dla każdego z 4 mikrofonów w każdej z 1259 ramek

    # n_pairs, n_sources, n_frames = tdoa_list.shape
    # estimated_positions = np.zeros((n_sources, n_frames, 2))

    # dla kżdej ramki 2 pozycja (1259, 2, 2)
    est = np.zeros((tdoa_list.shape[2], tdoa_list.shape[1], 2))

    # łatwiej będie analizować dla kojedyńczych ramek
    for frame_idx in range(tdoa_list.shape[2]):
        print(f"\r{frame_idx}", end='')
        # a = tdoa_list[:, :, frame_idx]
        # print(a.shape)
        # print(a)
        # TODO error_map len = src_number
        error_map = [np.zeros_like(X), np.zeros_like(X)]
        # teraz po każdej parze mikrofonow
        for mic_idx in range(tdoa_list.shape[0]):
            mic1_idx = mic_pairs[mic_idx][0] # numer 1 mikrofon
            mic2_idx = mic_pairs[mic_idx][1] # numer 2 mikrofonu
            mic1_pos = mic_positions[mic1_idx]
            mic2_pos = mic_positions[mic2_idx]
            # print(mic1_pos, mic2_pos)

            d1x = X - mic1_pos[0]
            d1y = Y - mic1_pos[1]
            d2x = X - mic2_pos[0]
            d2y = Y - mic2_pos[1]

            d1 = np.sqrt(d1x * d1x + d1y * d1y)
            d2 = np.sqrt(d2x * d2x + d2y * d2y)

            H = d1 - d2

            # dla każdego potencjalnego źródła
            for src_idx in range(tdoa_list.shape[1]):
                tdoa = tdoa_list[mic_idx, src_idx, frame_idx]
                delta_d = c * tdoa
                H1 = H - delta_d
                error_map[src_idx] += H1 * H1

        error_map = np.array(error_map)

        for i in range(len(error_map)):
            min_idx = np.unravel_index(np.argmin(error_map[i]), error_map[i].shape)
            est_x = X[min_idx]
            est_y = Y[min_idx]
            est[frame_idx, i] = [est_x, est_y]

    return est

def animateTrajectories(pos, interval=20, tail_length=50):
    n_frames = pos.shape[0]
    n_sources = pos.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))

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

        time_text.set_text(f'Klatka: {frame}')
        return lines + points + [time_text]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True, repeat=False)

    plt.show()
    return ani

def main():
    # DATA
    timeSamplesGenerator, numberOfChannels, sampleRate = timeSamplesGeneratorFromWavFile(wavFiles[0], n = n)

    ccs = [
        [],
        [],
        [],
        []
    ]

    # GCC-PHAT
    for frame, block in enumerate(timeSamplesGenerator):
        print(f"\r{frame}", end = "")

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

    # np.save('./data/ccs_96.npy', ccs)
    # exit(0)

    # plotHeatmap2(ccs[0]) # TEST

    # VITERBI

    numberOfLines = 2
    beta = 2.0
    emissionWeight = 5.0
    removeRadius = 3

    allPaths = []

    for i in range(ccs.shape[0]):
        heatmap = ccs[i]
        heatmap = heatmap.T

        paths = calculatePathsVA(
            heatmap = heatmap,
            numberOfLines = numberOfLines,
            beta = beta,
            emissionWeight = emissionWeight,
            removeRadius = removeRadius
        )

        print(paths.shape)

        allPaths.append(paths)

        # TEST
        # plt.figure(figsize=(8, 5))
        # plt.imshow(heatmap, cmap='gray', aspect='auto', origin='lower')
        # for p in paths:
        #     plt.plot(p, linewidth=2)
        # plt.tight_layout()
        # plt.show()

    allPaths = np.array(allPaths)

    # np.save('data/allPaths_tda.npy', allPaths)
    # exit(0)

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


    print(allPaths.shape)

    # zamiana na sekundy
    allPaths = (allPaths - delayRange) / sampleRate

    pos = hyperbolaContours(allPaths, microphonePairs, microphonePositions, X, Y)

    print(pos)
    # (1259, 2, 2)
    print(pos.shape)

    # pos = pos[:,0]
    #
    # plt.figure(figsize=(8, 5))
    # sc1 = plt.scatter(pos[:, 0, 0], pos[:, 0, 1], c=np.arange(pos.shape[0]), cmap='viridis')
    # sc2 = plt.scatter(pos[:, 1, 0], pos[:, 1, 1], c=np.arange(pos.shape[0]), cmap='plasma')
    # plt.colorbar(sc1)
    # plt.show()

    ani = animateTrajectories(pos)
    # ani.save('./hiperbole/trajektoria.mp4', writer='ffmpeg', fps=30)

if __name__ == '__main__':
    t1 = time.thread_time()
    main()
    t2 = time.thread_time()
    print(f'Time: {t2 - t1} s')