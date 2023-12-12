#!/bin/bash
> logs/res.log
for ((i = 30; i <= 200; i += 10))
do 
    for ((j = 5; j <= i; j += 5)) 
    do python main_classifier.py $i $j
    done
done