This repository is currently under active cleaning.

# Installation steps
1. Create conda env
```
conda create -n dlp python=3.10.8; conda activate dlp
```
2. Install PyTorch
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
3. Install packages required by our package
```
pip install -r requirements.txt
```
4. Install BabyAI
```
pip install blosc; cd babyai; pip install -e .; cd ..
```
5. Install gym-minigrid
```
cd gym-minigrid; pip install -e.; cd ..
```
6. Install Accelerate
```
cd v0.13.2/accelerate-0.13.2; pip install -e .; cd ../..
```
7. Install Lamorel
```
cd language-models-for-rl/lamorel; pip install -e .; cd ../..
```
