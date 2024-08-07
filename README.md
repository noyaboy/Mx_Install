# Mx_Install
## Dependency Setup
Create an new conda virtual environment
```
conda create -n unfold_mx python=3.9.12 -y
conda activate unfold_mx
```
Install environment packages <br>
(If meet problem, please try panda==1.3.5, latest numpy)
```
conda install cudatoolkit=11.3 -c conda-forge
conda install cudatoolkit-dev=11.3 -c conda-forge
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install packaging
pip install ninja
pip install pytest
conda install gxx_linux-64=9.3.0
pip install numpy==1.21.6
conda install pandas
```
Set variables:
```
conda env config vars set PYTHONPATH="$PYTHONPATH:/home1/science103555/unfold_mx/microxcaling/"
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
Export environment:
```
cd ~
conda env export > unfold_mx.yml
```
Another way to install:<br>
https://hackmd.io/pli0jSzZQ8iOEwYJPUZtIg?fbclid=IwY2xjawEZbgdleHRuA2FlbQIxMAABHah702ZyNrqd7-CWViNX6EhIznoYSHqrL5S4USTMbIgHIhYb8s9qI4hA2w_aem_w6XCRrkaNN7M8J1An6JF1g
