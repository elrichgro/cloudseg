#!/bin/bash

echo "Copying files into ramdisk"
ROOT="/mnt/ds3lab-scratch/llodrant"

# fix permission that I ... up when I create the dataset
find $ROOT/data -type d -print0 | xargs -0 chmod 0755
find $ROOT/data -type f -print0 | xargs -0 chmod 0644

rm -rf /dev/shm/llodrant
rm -f $ROOT/tmp

mkdir -p /dev/shm/llodrant
ln -s /dev/shm/llodrant $ROOT/tmp

cp -r $ROOT/data/datasets/$1 $ROOT/tmp/dataset

echo "Starting training"
PYTHONPATH=$PYTHONPATH:$ROOT python irccam/training/train.py -c config-spaceml2.json

rm -r /dev/shm/llodrant
rm $ROOT/tmp
