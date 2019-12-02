# mil-seminar2019
Network Architecture Search System

## Files
- ```run.sh```: Batch script for reedbush.
- ```src/main.py```: Main script.
- ```src/task.py```: Defines task class.
- ```src/preprocess.py```: Defines preprocess selection method.
- ```src/nas.py```: Defines network architecture optimization algorithm.
- ```src/hyperparam.py```: Defines hyperparameter class.

## Usage
To run on lab servers
```
python main.py --task cifer100 --gpu 0
```

To run batch script on reedbush
```
qsub run.sh
rbstat
```
