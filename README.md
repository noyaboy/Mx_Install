# Mx_Install
## Dependency Setup
Create an new conda virtual environment
```
conda create -n unfold_mx python=3.9.12 -y
conda activate unfold_mx
```
Install environment packages
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```
