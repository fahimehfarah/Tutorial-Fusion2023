#!/bin/bash

echo Today is $(date) and is a beautiful day
echo Powered by Luca Zelioli

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHONPATH=/home/"$USER"/Documents/Tutorial-Fusion2023

conda activate fusion2023

echo Write target height default 512
read -r h

echo Write target width default 928
read -r w

echo Write number of epochs
read -r e

echo Select type early or middle or late fusion
read -r type

if [ "$type" = "early" ]
  then
    echo Start Early fusion
    python early_fusion.py -height "${h}" -width "${w}" -epochs "${e}"
elif [ "$type" = "middle" ]
  then
    echo Start Middle fusion
    python middle_fusion.py -height "${h}" -width "${w}" -epochs "${e}"
else
    echo Start Late fusion
    python late_fusion.py -height "${h}" -width "${w}" -epochs "${e}"
fi