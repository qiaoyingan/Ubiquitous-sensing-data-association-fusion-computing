#!/bin/bash
log_file='logs/res.log'
> $log_file
for ((i = 1; i <= 11; i += 1))
do python main_classifier.py 100 20 $i >> $log_file
done