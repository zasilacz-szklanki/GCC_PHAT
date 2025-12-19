import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import acoular as ac
from scipy.io import wavfile

ac.config.global_caching = 'none'

FPS = 30
frg_span = 0.2

coords = np.load("./output/wall1_positions.npy")
print("coords shape:", coords.shape)

# sampleRate, wavData = wavfile.read('./audio/white_noise_1sec.wav')
# sampleRate, wavData = wavfile.read('./audio/WhiteNoiseRound_8ch_96000Hz_24bit.wav')
sampleRate, wavData = wavfile.read('./audio/wall1.wav')

print(f"Fs={sampleRate}, shape={wavData.shape}")

x = wavData[:, 0].astype(float)
y = wavData[:, 3].astype(float)

mg = ac.MicGeom(file='ring8_capstone_wall.xml')
rg = ac.RectGrid(
    x_min=-2, x_max=+2,
    y_min=-2, y_max=+2,
    z=1,
    increment=0.1,
)

timeSamples = ac.TimeSamples(data=wavData, sample_freq=sampleRate)
st = ac.SteeringVector(grid=rg, mics=mg, steer_type='classic')

frames_count = int(timeSamples.num_samples / timeSamples.sample_freq * FPS)
frame_length = int(timeSamples.sample_freq / FPS)

print("frames_length = ", frame_length)

def mapIndexToRange(i, num, v_min, v_max):
    step = (v_max - v_min) / (num - 1)
    return v_min + (i * step)

def DrawLineGraph(arr, x0 = None):
    arr = np.array(arr)
    x = np.arange(arr.shape[0]) + 1
    y = arr

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linestyle='-', color='green')

    if x0 is not None:
        plt.axvline(x0, color='red', linestyle='--', linewidth=1.5)

    plt.xlabel("Iteracja")
    plt.ylabel("Wartość")
    plt.title("Średnia odlełość cząsteczek od centroidu")
    plt.grid(True)
    plt.show()

ccs = []
cc_maxs = []

def gcc_phat(xseg, yseg, fs):
    n = int(2**np.ceil(np.log2(len(xseg) + len(yseg))))
    # n = 8192
    X = np.fft.rfft(xseg, n=n)
    Y = np.fft.rfft(yseg, n=n)

    # TODO różnica faz
    R = X * np.conj(Y)
    R = np.exp(1j * np.angle(R))
    # R /= (np.abs(R) + 1e-12)

    cc = np.fft.ifft(R, n=n)
    cc = np.fft.ifftshift(cc)
    ccs.append(cc.real)

    cc_max = np.argmax(cc.real)
    cc_maxs.append(cc_max)

    # DrawLineGraph(cc.real, x0 = cc_max)

    tau = cc_max / fs
    return tau


generator = timeSamples.result(num=frame_length)

frames = []
rg_extent = rg.extent

t_start = time.thread_time()
for j, block in enumerate(generator):
    print(f'\r{j}/{frames_count}', end = '')
    if j >= frames_count:
        break

    block_ts = ac.TimeSamples(data=block, sample_freq=sampleRate)

    bf_block = ac.BeamformerTime(source=block_ts, steer=st)

    block_map = next(bf_block.result(frame_length))
    r = np.sum(block_map**2, axis=0)
    r = r.reshape(rg.shape)

    p = np.unravel_index(np.argmax(r), r.shape)
    px = mapIndexToRange(p[0], r.shape[0], rg_extent[0], rg_extent[1])
    py = mapIndexToRange(p[1], r.shape[1], rg_extent[2], rg_extent[3])

    frg = ac.RectGrid(
        x_min=px - frg_span,
        x_max=px + frg_span,
        y_min=py - frg_span,
        y_max=py + frg_span,
        z=1,
        increment=0.1,
    )
    fst = ac.SteeringVector(grid=frg, mics=mg, steer_type='classic')
    fbf_block = ac.BeamformerTime(source=block_ts, steer=fst)
    fblock_map = next(fbf_block.result(frame_length))
    fr = np.sum(fblock_map**2, axis=0)
    fr = fr.reshape(frg.shape)

    fp = np.unravel_index(np.argmax(fr), fr.shape)
    frg_extent = frg.extent
    fpx = mapIndexToRange(fp[0], fr.shape[0], frg_extent[0], frg_extent[1])
    fpy = mapIndexToRange(fp[1], fr.shape[1], frg_extent[2], frg_extent[3])

    xseg = block_ts.data[:, 0]
    yseg = block_ts.data[:, 3]
    # xseg = x[j*frame_length:(j+1)*frame_length]
    # yseg = y[j*frame_length:(j+1)*frame_length]
    tau = gcc_phat(xseg, yseg, sampleRate)
    t_cur = j / FPS

    idx = int(np.clip(j, 0, coords.shape[0]-1))
    cpx, cpy = coords[idx]

    frames.append((r, (px, py), fr, (fpx, fpy), tau, t_cur, (cpx, cpy)))

t_end = time.thread_time()

print('')

print(f'Czas wykonania przygotowania ramek: {t_end - t_start} s')
print(f'Liczba ramek: {len(frames)}')

fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])


ax_map = fig.add_subplot(gs[0,0])
ax_map.set_title("Beamformer 1")
ax_map.clear()
im_map = ax_map.imshow(np.zeros(rg.shape).T, extent=rg.extent, origin="lower")
pt_map, = ax_map.plot([], [], 'r+', markersize=10)
ann_map = ax_map.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", color='white')


ax_fmap = fig.add_subplot(gs[0,1])
ax_fmap.set_title("Beamformer 2")
ax_fmap.clear()
im_fmap = ax_fmap.imshow(np.zeros((10,10)).T, extent=(0,1,0,1), origin="lower")

pt_fmap, = ax_fmap.plot([], [], 'r+', markersize=10)
ann_fmap = ax_fmap.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", color='white')


ax_angle = fig.add_subplot(gs[1,0])
ax_angle.set_title("Kąt (GCC-PHAT) w czasie")
line_angle, = ax_angle.plot([], [], 'r-')
ax_angle.set_xlim(0, frames[-1][5] if frames else 1.0)
# ax_angle.set_ylim(-90, 90)
ax_angle.set_xlabel("Czas")
ax_angle.set_ylabel("Kat")
ann_angle = ax_angle.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", color='black')


ax_pos = fig.add_subplot(gs[1,1])
ax_pos.set_title("Pozycja z pliku npy")
ax_pos.set_aspect('equal')
ax_pos.set_xlim(-2, 2)
ax_pos.set_ylim(-2, 2)
pt_pos, = ax_pos.plot([], [], 'go')
ann_pos = ax_pos.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                          arrowprops=dict(arrowstyle="->"))


def plotHeatmap(matrix, cc_maxs = None, name ='gp5'):
    plt.figure(figsize=(20, 20))
    plt.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')

    if cc_maxs is not None:
        line = np.array(cc_maxs)
        y = np.arange(line.shape[0])
        plt.plot(line, y, color='red', linewidth=2)

    plt.xlabel("Kolumny")
    plt.ylabel("Wiersze")
    plt.title(f"Heatmapa macierzy {matrix.shape}")
    plt.tight_layout()
    plt.savefig(f'test/{name}_wall1_ts.svg', format="svg")
    plt.show()


print('=======')
ccs = np.array(ccs)
print(ccs.shape)
# print(ccs)
###############
plotHeatmap(ccs, cc_maxs = cc_maxs, name ='gp5_full')
plotHeatmap(ccs[:, 3800:4201], cc_maxs = np.clip(np.array(cc_maxs) - 3800, 0, 400), name ='gp5_sub')
print('=======')

'''
def init():
    im_map.set_data(np.zeros(rg.shape).T)
    pt_map.set_data([], [])
    ann_map.set_text("")
    im_fmap.set_data(np.zeros((10,10)).T)
    pt_fmap.set_data([], [])
    ann_fmap.set_text("")
    # kąt
    line_angle.set_data([], [])
    # pozycja
    pt_pos.set_data([], [])
    ann_pos.set_text("")

    ann_angle.set_text("")

    return im_map, pt_map, ann_map, im_fmap, pt_fmap, ann_fmap, line_angle, pt_pos, ann_pos


def update(frame_idx):
    r, (px, py), fr, (fpx, fpy), theta, t_cur, (cpx, cpy) = frames[frame_idx]

    im_map.set_data(r.T)
    im_map.set_extent(rg.extent)
    pt_map.set_data([px], [py])
    ann_map.xy = (px, py)
    ann_map.set_text(f"({px:.2f}, {py:.2f})")

    frg_extent = (px - frg_span, px + frg_span, py - frg_span, py + frg_span)
    im_fmap.set_data(fr.T)
    im_fmap.set_extent(frg_extent)
    pt_fmap.set_data([fpx], [fpy])
    ann_fmap.xy = (fpx, fpy)
    ann_fmap.set_text(f"({fpx:.3f}, {fpy:.3f})")

    times = [frames[i][5] for i in range(frame_idx + 1)]
    angles = [frames[i][4] for i in range(frame_idx + 1)]
    line_angle.set_data(times, angles)

    ann_angle.xy = (t_cur, theta)
    ann_angle.set_text(f"{theta:.2f}°")

    pt_pos.set_data([cpx], [cpy])
    ann_pos.xy = (cpx, cpy)
    ann_pos.set_text(f"({cpx:.2f}, {cpy:.2f})")

    print(f"Klatka {frame_idx+1}/{len(frames)}  t={t_cur:.2f}s  theta={theta:.2f}°", flush=True, file=sys.stderr)

    return im_map, pt_map, ann_map, im_fmap, pt_fmap, ann_fmap, line_angle, pt_pos, ann_pos, ann_angle


ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              init_func=init, interval=1000//FPS, repeat=True)

plt.tight_layout()

ani.save("./output/beamforming_gccphat_gp5.mp4", writer="ffmpeg", fps=FPS)

plt.show()
'''