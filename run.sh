#!/bin/sh

#PBS -q l-small
#PBS -l select=1:mpiprocs=1:ompthreads=1
#PBS -W group_list=gk75
#PBS -l walltime=24:00:00

LUSTRE=/lustre/gk**/k*****

PBS_O_WORKDIR=$LUSTRE/mil-seminar2019/src

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module load cuda10/10.0.130

export PYENV_ROOT=$LUSTRE/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
export PYTHONUSERBASE=$LUSTRE/python_packages
export PYTHONPATH=$LUSTRE

python main.py --task cifer100 --gpu 0
