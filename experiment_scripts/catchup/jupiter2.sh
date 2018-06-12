python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d lis -g 2 -m unet -p schop
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 2 -m segnet -p mnat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 2 -m fcn -p snat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 2 -m fcn -p schop
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d psd -g 2 -m segnet -p schop
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d psd -g 2 -m fcn -p lnat
