# GBIM

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