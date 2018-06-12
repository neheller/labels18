import numpy as np
from pathlib import Path
import sys

psd_dir = Path("/home/helle246/data/PSD/validation")
lis_dir = Path("/home/helle246/data/LiS/validation")
viz_dir = Path(sys.argv[0]).parents[0]

num_to_shuffle = 5

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

    XY = load_bundles(num_to_shuffle, sparse, glob, xpfx, pth)
    np.random.shuffle(XY)
    destx = destination / "xviz.npy"
    desty = destination / "yviz.npy"
    np.save(str(destx), XY[0:20,:,:,[0]])
    np.save(str(desty), XY[0:20,:,:,[1,2]])
