#!/bin/bash


for folder in $1/*
do
    for file in "$folder"/*.mp4
	
    do
		echo "${file%.mp4}"
        if [[ ! -d "${file%.mp4*}" ]]; then
            mkdir -p "${file%.mp4*}"
        fi
        ffmpeg -i "$file" -vf fps=5 "${file%.mp4*}"/%05d.jpg
        rm "$file"
    done
done
