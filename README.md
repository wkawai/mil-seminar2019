# mil-seminar2019
Network Architecture Search System

## Files
- ```run.sh```: batch script for reedbush.
- ```src/main.py```: main script.
- ```src/task.py```: defines task class.
- ```src/preprocess.py```: defines preprocess selection method.
- ```src/nas.py```: defines network architecture optimization algorithm.
- ```src/hyperparam.py```: defines hyperparameter class.

## Usage
To run on lab servers,
```
python main.py --task cifer100 --gpu 0
```

To run batch script on reedbush,
```
qsub run.sh
rbstat
```
