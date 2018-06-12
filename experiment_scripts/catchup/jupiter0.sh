python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d lis -g 0 -m unet -p snat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 0 -m segnet -p snat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 0 -m segnet -p schop
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup1 -d lis -g 0 -m fcn -p mnat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d psd -g 0 -m segnet -p mnat
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n catchup0 -d psd -g 0 -m fcn -p snat
