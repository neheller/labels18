from pathlib import Path
import numpy as np
import sys
import csv

log_root = Path("/home/helle246/data/labels18_models")

def get_performance(model, perturb, dataset):
    ret = []
    for p in log_root.glob("**/*.csv"):
        run = p.parent.name
        if ((model in run) and (perturb in run) and (dataset in run)):
            with p.open('r') as f:
                r = csv.reader(f)
                try:
                    r.__next__()
                except StopIteration:
                    print(str(p))
                    continue
                val_dices = [float(row[7]) for row in r]
            ret = ret + [max(val_dices)]
    # print(model, perturb, dataset, len(ret))
    return ret

if __name__ == "__main__":
    model = sys.argv[1]
    perturb = sys.argv[2]
    dataset = sys.argv[3]
    print(get_performance(model, perturb, dataset))
