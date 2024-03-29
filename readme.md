# GBIM
This repository contains the source code of the AAAI-2024 paper "[Graph Bayesian Optimization for Multiplex Influence Maximization](https://arxiv.org/abs/2403.18866)"

Simply run the following command to 
test the GBIM in Multi-IC of the lastfm dataset with k=5.   
```
python main.py
```

more data are in data.zip, you can download it from [here](https://drive.google.com/file/d/12-vouHWJIu5SkWnKRMOSEQa1x32nSMkS/view?usp=drive_link) and unzip it into ./data


To use these data, you need to first run
```
python datasample.py -d ciao -dm LT -k 5
```
to generate the initial training data D,

then run
```
python main.py -d ciao -dm LT -k 5
```
ciao is one dataset in [ciao, epinions, delicous, lastfm]

dm [LT, IC]

k is the seed set size

Please cite our paper if it is helpful to your work.
```latex
@inproceedings{yuan2024graph,
  title={Graph Bayesian Optimization for Multiplex Influence Maximization},
  author={Yuan, Zirui and Shao, Minglai and Chen, Zhiqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={20},
  pages={22475--22483},
  year={2024}
}
```
