import sys
from pathlib import Path

def get_runs():
    # Necessary Runs
    models = ["unet", "fcn", "segnet"]
    datasets = ["psd", "lis"]
    perturbations = ["srnd", "mrnd", "lrnd"]
    return models, datasets, perturbations

def get_runs_by_hand():
    return [
        ("unet", "lis", "snat", "catchup0"),
        ("unet", "lis", "mnat", "catchup0"),
        ("unet", "lis", "schop", "catchup0"),
        ("segnet", "lis", "snat", "catchup0"),
        ("segnet", "lis", "snat", "catchup1"),
        ("segnet", "lis", "mnat", "catchup0"),
        ("segnet", "lis", "mnat", "catchup1"),
        ("segnet", "lis", "schop", "catchup0"),
        ("segnet", "lis", "schop", "catchup1"),
        ("fcn", "lis", "snat", "catchup0"),
        ("fcn", "lis", "snat", "catchup1"),
        ("fcn", "lis", "mnat", "catchup0"),
        ("fcn", "lis", "mnat", "catchup1"),
        ("fcn", "lis", "schop", "catchup0"),
        ("fcn", "lis", "schop", "catchup1"),
        ("segnet", "psd", "snat", "catchup0"),
        ("segnet", "psd", "mnat", "catchup0"),
        ("segnet", "psd", "lnat", "catchup0"),
        ("segnet", "psd", "schop", "catchup0"),
        ("segnet", "psd", "mchop", "catchup0"),
        ("fcn", "psd", "snat", "catchup0"),
        ("fcn", "psd", "mnat", "catchup0"),
        ("fcn", "psd", "lnat", "catchup0"),
        ("fcn", "psd", "lchop", "catchup0")
    ]

def get_assets():
    # Available Resources
    gpus = ["jupiter0", "jupiter1", "jupiter2", "jupiter3", "jinx0", "jinx1", "jinx2"]
    return gpus

def get_access_info():
    # Who has access to what
    gpu_dataset_access = {
        "jupiter0": ["psd", "lis"],
        "jupiter1": ["psd", "lis"],
        "jupiter2": ["psd", "lis"],
        "jupiter3": ["psd", "lis"],
        "rocinante0": ["lis"],
        "jinx0": ["lis"],
        "jinx1": ["lis"],
        "jinx2": ["lis"],
    }
    return gpu_dataset_access

class Experiment(object):

    def __init__(self, model, dataset, perturbation, name):
        self.model = model
        self.dataset = dataset
        self.perturbation = perturbation
        self.name = name
        self.gpu = -1

    def __str__(self):
        runner = "/home/helle246/code/repos/Perturbation-Networks/model_runner.py"
        formatted_call = "python3 %s -n %s -d %s -g %d -m %s -p %s"
        return formatted_call % (
            runner, self.name, self.dataset,
            self.gpu, self.model, self.perturbation
        )

class LoadBalancer(object):

    def __init__(self):
        self.gpus = get_assets()
        self.gpu_dataset_access = get_access_info()
        self.assignments = {gpu: [] for gpu in self.gpus}

    def assign(self, exp):
        candidates = [gpu for gpu in self.gpus
                        if exp.dataset in self.gpu_dataset_access[gpu]]
        candidate_loads = [len(self.assignments[c]) for c in candidates]
        argmin = candidate_loads.index(min(candidate_loads))
        exp.gpu = int(candidates[argmin][-1])
        self.assignments[candidates[argmin]].append(exp)

    def get_script(self, gpu):
        ret = ""
        for cmd in self.assignments[gpu]:
            ret = ret + str(cmd) + "\n"
        return ret

    def __str__(self):
        ret = ""
        for gpu in self.assignments:
            ret = ret + "********** " + gpu + " **********\n"
            ret = ret + self.get_script(gpu)
        return ret

    def write_scripts(self, destination):
        for gpu in self.assignments:
            script = destination / (gpu + ".sh")
            with script.open('w') as f:
                f.write(self.get_script(gpu))

if __name__ == "__main__":
    destination = Path(sys.argv[1])
    redundancy = int(sys.argv[3])
    models, datasets, perturbations = get_runs()
    experiments = [
        Experiment(model, dataset, perturb, sys.argv[2]+str(i))
        for i in range(0, redundancy)
        for model in models
        for dataset in datasets
        for perturb in perturbations
    ]
    # quads = get_runs_by_hand()
    # experiments = [
    #     Experiment(model, dataset, perturb, name)
    #     for (model, dataset, perturb, name) in quads
    # ]
    # Assign each job to a gpu
    lb = LoadBalancer()
    for exp in experiments:
        lb.assign(exp)

    # print(lb)
    lb.write_scripts(destination)
