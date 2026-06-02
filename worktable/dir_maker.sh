#!/bin/bash

parent_dir="/home/tahia/Documents/MSC/derivatives/"
touch "./dl_log.txt"

for subj in {00..10}; do
	subj_num="sub-MSC$subj/"

	for sess in {00..10}; do
		run_num="ses-func${sess}/"
		mkdir ${parent_dir}${subj_num}${run_num}datasets

	done
done

