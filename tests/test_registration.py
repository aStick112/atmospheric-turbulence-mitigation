import math
from time import perf_counter
from datetime import datetime
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage
import atmospheric_turbulence_mitigation.registration as reg

PATTERN_PATH = "/resources/OTIS_PNG_Gray/Fixed Patterns/Pattern15"
GT = skimage.io.imread(f"{PATTERN_PATH}/GT/Pattern15_GT.png")


def main():
    im_seq = []
    for path in glob.glob(f"{PATTERN_PATH}/*.png"):
        im_seq.append(plt.imread(path))

    im_seq = np.array(im_seq)

    temporal_avg = np.mean(im_seq, axis=0)
    temporal_med = np.median(im_seq, axis=0)
    temporal_std = np.std(im_seq, axis=0)

    iterations = 3

    t_i = perf_counter()
    registered = reg.stabilize(
        im_seq,
        iterations=iterations,
        reference_filter=lambda seq: np.mean(seq, axis=0)
    )
    t_f = perf_counter()
    completion_time = t_f - t_i

    print(f"Completion time: {round(completion_time, 3)} seconds")

    f, ax = plt.subplots(2, 3)

    plt.suptitle(f"Completion time: {round(completion_time, 3)} seconds")

    ax[0][0].imshow(GT, cmap=mpl.colormaps["Greys_r"])
    ax[0][0].set(
        title="Ground Truth",
        xticks=[], yticks=[]
    )

    ax[0][1].imshow(im_seq[0], cmap=mpl.colormaps["Greys_r"])
    ax[0][1].set(
        title="Turbulent Sample",
        xticks=[], yticks=[]
    )

    ax[0][2].imshow(temporal_avg, cmap=mpl.colormaps["Greys_r"])
    ax[0][2].set(
        title="Temporal Average",
        xticks=[], yticks=[]
    )

    ax[1][0].imshow(registered[0], cmap=mpl.colormaps["Greys_r"])
    ax[1][0].set(
        title=f"(Iters = 1)",
        xticks=[], yticks=[]
    )

    ax[1][1].imshow(registered[min(math.ceil(len(registered) / 2), len(registered))], cmap=mpl.colormaps["Greys_r"])
    ax[1][1].set(
        title=f"(Iters = {math.ceil(len(registered) / 2)})",
        xticks=[], yticks=[]
    )

    ax[1][2].imshow(registered[-1], cmap=mpl.colormaps["Greys_r"])
    ax[1][2].set(
        title=f"(Iters = {iterations})",
        xticks=[], yticks=[]
    )

    plt.savefig(f"outputs/output@{datetime.now().strftime('%Y-%m-%d %H%M%S')}")
    plt.show()


if __name__ == '__main__':
    main()
