#!/bin/bash


> train.list
> test.list
COUNT=-1
for folder in $1/*
do
    COUNT=$[$COUNT + 1]
    for imagesFolder in "$folder"/*
    do
        if (( $COUNT%5 != 0 )); then
            echo "$imagesFolder" $COUNT >> train.list
        else
            echo "$imagesFolder" $COUNT >> test.list
        fi        
    done
done
