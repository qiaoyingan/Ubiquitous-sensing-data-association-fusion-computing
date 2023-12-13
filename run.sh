#!/bin/bash
log_file='logs/res.log'
> $log_file
for ((i = 120; i <= 400; i += 10))
do 
    for ((j = 5; j <= 40; j += 5)) 
    do python main_classifier.py $i $j >> $log_file
    done
done