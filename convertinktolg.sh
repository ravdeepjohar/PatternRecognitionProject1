#!/bin/bash

FILENAME="AllEM_part4_TRAIN_all.txt"
count=0
cat $FILENAME | while read LINE
do
        temp=${LINE##*/}
	temp2=${temp%.inkml}
	lgname="all_lg/"$temp2".lg"
	perl ../Downloads/CROHMELib/bin/crohme2lg.pl $LINE $lgname
done

