#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOmultitask/ICbaseline 

python3 DoGoAic.py CC
