# Mx_Install
## Dependency Setup
Create an new conda virtual environment
```
conda create -n unfold_mx python=3.9.12 -y
conda activate unfold_mx
```
Install environment packages
```
conda install cudatoolkit=11.3 -c conda-forge
conda install cudatoolkit-dev=11.3 -c conda-forge
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda env config vars set PYTHONPATH="$PYTHONPATH:/home1/zks878/mx/microxcaling/"

pip install packaging
pip install ninja
pip install pytest
conda install gxx_linux-64=9.3.0
pip install numpy==1.21.6
```
Check version:
```
python3
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
exit()
```
