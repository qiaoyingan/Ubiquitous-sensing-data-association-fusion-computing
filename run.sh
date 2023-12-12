#!/bin/bash
> res.log
for ((i = 30; i <= 200; i += 10))
do 
    for ((j = 5; j <= i; j += 5)) 
    do python main_classifier.py $i $j
    done
done
# python main_classifier.py 100 20