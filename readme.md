# GBIM
This repository contains the source code of the AAAI-2024 paper "Graph Bayesian Optimization for Multiplex Influence Maximization"

Simply run the following command to 
test the GBIM in Multi-IC of the lastfm dataset with k=5.   
```
python main.py
```

more data are in data.zip, you can unzip it into ./data
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
@article{Yuan_Shao_Chen_2024,
title={Graph Bayesian Optimization for Multiplex Influence Maximization},
volume={38},
url={https://ojs.aaai.org/index.php/AAAI/article/view/30255},
DOI={10.1609/aaai.v38i20.30255}, number={20},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Yuan, Zirui and Shao, Minglai and Chen, Zhiqian},
year={2024},
month={Mar.},
pages={22475-22483}
}
```
