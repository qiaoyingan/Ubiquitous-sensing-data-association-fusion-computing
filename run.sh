#!/bin/bash
> logs/res.log
for ((i = 120; i <= 400; i += 10))
do 
    for ((j = 5; j <= 40; j += 5)) 
    do python main_classifier.py $i $j
    done
done