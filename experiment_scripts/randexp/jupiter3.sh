python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp0 -d lis -g 3 -m unet -p srnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp0 -d lis -g 3 -m fcn -p mrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp0 -d lis -g 3 -m segnet -p lrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp1 -d psd -g 3 -m fcn -p srnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp1 -d psd -g 3 -m segnet -p mrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp2 -d psd -g 3 -m unet -p lrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp2 -d lis -g 3 -m fcn -p srnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp2 -d lis -g 3 -m segnet -p mrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp3 -d lis -g 3 -m unet -p lrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp3 -d psd -g 3 -m segnet -p srnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp4 -d psd -g 3 -m unet -p mrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp4 -d psd -g 3 -m fcn -p lrnd
python3 /home/helle246/code/repos/Perturbation-Networks/model_runner.py -n randexp4 -d lis -g 3 -m segnet -p srnd
