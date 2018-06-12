# LABELS 2018 Code

A study of how erroneous training data affects the performance of deep learning
systems for semantic segmentation

## Usage

```
usage: model_runner.py [-h] [-n NEW] [-e EXISTING] [-b BATCH_SIZE]
                       [-d DATASET] [-p PERTURBATION] [-g GPU_INDEX]
                       [-m MODEL] [-t [TESTING]] [-v [VISUALIZE]]

Run a training or testing round for our Labels 2018 Submission

optional arguments:
  -h, --help            show this help message and exit
  -n NEW, --new NEW     A name for a new training session from random
                        initialization
  -e EXISTING, --existing EXISTING
                        A name for an existing training session to continue
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to use during training
  -d DATASET, --dataset DATASET
                        The dataset to train/test on either lis or psd
  -p PERTURBATION, --perturbation PERTURBATION
                        The type of perturbations to do to the training data
  -g GPU_INDEX, --gpu GPU_INDEX
                        The GPU to limit this training run to.
  -m MODEL, --model MODEL
                        The architecture to use for this round
  -t [TESTING], --testing [TESTING]
                        Whether this round will be a testing round
  -v [VISUALIZE], --visualize [VISUALIZE]
                        Whether to visualize

```

## Directory Structure

**callbacks**

A directory of callbacks used during training

* **TopNSaver**: Saves top N models during training according to specified metric
* **VizPreds**: Visualizes predictions made by model during training

**Models**

A directory of files which define and compile the models for training

**viz**

Put an x,y .npy pair here and it will visualize predictions on this data during
training

## Acknowledgements
* UNet Implementation [Courtesy of divamgupta](https://github.com/divamgupta/image-segmentation-keras)
* FCN8 Implementation [Courtesy of divamgupta](https://github.com/divamgupta/image-segmentation-keras)
* SegNet Implementation [Courtesy of ykamikawa](https://github.com/ykamikawa/SegNet)
