import matplotlib.pyplot as plt
from get_performance import get_performance
import numpy as np
import sys

def get_marker_and_color(pert):
    if (pert == "control"):
        return '.', 'blue'
    if (pert in ["snat", "schop", "srnd"]):
        return 'x', 'cyan'
    if (pert in ["mnat", "mchop", "mrnd"]):
        return '*', 'orange'
    if (pert in ["lnat", "lchop", "lrnd"]):
        return '+', 'red'


pertsets = [
    ["control", "snat", "mnat", "lnat"],
    ["control", "schop", "mchop", "lchop"],
    ["control", "srnd", "mrnd", "lrnd"]
]
fig = plt.figure()
ax = fig.add_subplot(111)
i = 0
for model in ["unet", "segnet", "fcn"]:
    i = i + 1
    for pertset in pertsets:
        i = i + 0.3
        for pert in pertset:
            ys = get_performance(model, pert, sys.argv[1])
            if (len(ys) <= 4):
                print(model, pert, len(ys))
            xs = [i for _ in ys]
            marker, color = get_marker_and_color(pert)
            cs = [color for _ in ys]
            if model=="unet" and (pert.endswith("chop") or (pert == "control" and "lchop" in pertset)):
                if pert.startswith('c'):
                    label="No Perturbation (1.0)"
                if pert.startswith('s'):
                    label="Small Perturbation (0.95)"
                if pert.startswith('m'):
                    label="Moderate Perturbation (0.90)"
                if pert.startswith('l'):
                    label="Large Perturbation (0.85)"
                ax.scatter(xs, ys, marker=marker, color=cs, label=label)
            else:
                ax.scatter(xs, ys, marker=marker, color=cs)
plt.xticks(
    [1.30, 1.60, 1.90,  3.20, 3.50, 3.80,  5.10, 5.40, 5.70],
    ["U-Net Natural", "U-Net Choppy", "U-Net Random",
    "SegNet Natural", "SegNet Choppy", "SegNet Random",
    "FCN32 Natural", "FCN32 Choppy", "FCN32 Random"],
    rotation=45, ha="right")
if (sys.argv[1] == "lis"):
    plt.ylim((0.5,1.0))
else:
    plt.ylim((0.1,0.6))
plt.subplots_adjust(bottom=0.2)
ax.legend(loc=3)
ax.yaxis.grid(linestyle='--')
plt.ylabel("Dice Sorensen Score")
plt.title("Liver Segmentation Results")
plt.show()
