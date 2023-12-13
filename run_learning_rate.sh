#!/bin/bash
> logs/res.log
for ((i = 1; i <= 11; i += 1))
do python main_classifier.py 100 20 $i
done