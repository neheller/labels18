import numpy as np
import sys
from pathlib import Path

from preprocessing.tools import perturb




psd_dir = Path("/home/helle246/data/PSD/validation")
lis_dir = Path("/home/helle246/data/LiS/validation")
viz_dir = Path(sys.argv[0]).parents[0]

num_to_shuffle = 5
num_trials = 2

# Lots of redundant code here ... # TODO
def load_bundles(num_to_shuffle, sparse, glob, xpfx, pth):
    i = 0
    XY = np.zeros((num_to_shuffle*20, 512, 512, 3))
    for f in pth.glob(glob):
        if (i >= num_to_shuffle):
            return XY
        typ = f.name.split('-')[1]
        index = int(f.name.split('-')[-1].split('.')[0])
        XY[20*i:20*(i+1),:,:,[0]] = np.load(
            str(pth / ('%s-%s-%d.npy' % (xpfx, typ, index)))
        )
        if (sparse):
            XY[20*i:20*(i+1),:,:,[1]] = np.load(str(f))
            XY[20*i:20*(i+1),:,:,[2]] = 1 - XY[20*i:20*(i+1),:,:,[1]]
        else:
            XY[20*i:20*(i+1),:,:,[1,2]] = np.load(str(f))
        i = i + 1
    return XY

def dice(x, y):
    x = x[:,:,:,0]
    y = y[:,:,:,0]
    intersection = np.sum(np.logical_and(x,y))
    nx = np.sum(x)
    ny = np.sum(y)
    prec = intersection/nx
    recall = intersection/ny
    dce = 2*intersection/(nx + ny)
    return prec, recall, dce


if __name__ == "__main__":
    ds = sys.argv[1].lower()
    destination = viz_dir / ds
    if (ds == "lis"):
        pth = lis_dir
        glob = "y*ntl*lo*.npy"
        sparse = False
        xpfx = "x"
    elif (ds == "psd"):
        pth = psd_dir
        glob = "Y*ntl*.npy"
        sparse = True
        xpfx = "X"
    else:
        print("dataset: %s not supported" % ds)
        sys.exit()

    pert_modes = ["control", "schop", "mchop", "lchop", "snat", "mnat", "lnat", "srnd", "mrnd", "lrnd"]
    pert_modes = ["control", "srnd", "mrnd", "lrnd"]
    dices = {mode: [] for mode in pert_modes}
    precisions = {mode: [] for mode in pert_modes}
    recalls = {mode: [] for mode in pert_modes}
    XY = load_bundles(num_to_shuffle, sparse, glob, xpfx, pth)
    Y = XY[:,:,:,[1,2]]
    f = np.mean(Y[:,:,:,0])
    for i in range(0, num_trials):
        np.random.shuffle(Y)
        bndl = Y[0:20]
        for mode in pert_modes:
            perturbed = perturb(bndl, mode, ds, f)
            p,r,d = dice(perturbed, bndl)
            precisions[mode].append(p)
            recalls[mode].append(r)
            dices[mode].append(d)

    for key in dices:
        print(key, np.mean(dices[key]), np.mean(precisions[key]), np.mean(recalls[key]))
