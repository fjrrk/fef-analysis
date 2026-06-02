#!/bin/bash

parent_dir="/home/tahia/Documents/MSC/ds000224/"
touch "./dl_log.txt"

for subj in {00..10}; do
	subj_num="sub_MSC$subj/"

	# if [[ ! -f "${parent_dir}${subj_num}ses-struct01/"]]
	git annex get "${parent_dir}${subj_num}ses-struct01/anat/"
	git annex get "${parent_dir}${subj_num}ses-struct02/anat/"
	ls "${parent_dir}${subj_num}ses-struct01/anat/" >> "./dl_log.txt"
	ls "${parent_dir}${subj_num}ses-struct02/anat/" >> "./dl_log.txt"
	for sess in {00..10}; do
		run_num="ses-func${sess}/"
		echo "Running subject $subj session number $sess"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_task-memoryscenes_bold.nii.gz"
		git annex get "${parent_dir}${subj_num}func/${subj_num}_${run_num}_task-memoryscenes_bold.nii.gz"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_task-memoryscenes_bold.nii.gz" >> "./dl_log.txt"

		git annex get "${parent_dir}${subj_num}fmap/${subj_num}_${run_num}_magnitude1.nii.gz"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_magnitude1.nii.gz" >> "./dl_log.txt"

		git annex get "${parent_dir}${subj_num}fmap/${subj_num}_${run_num}_magnitude2.nii.gz"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_magnitude2.nii.gz" >> "./dl_log.txt"

		git annex get "${parent_dir}${subj_num}fmap/${subj_num}_${run_num}_phasediff.ni.gz"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_phasediff.ni.gz" >> "./dl_log.txt"

		git annex get "${parent_dir}${subj_num}func/${subj_num}_${run_num}_task-rest_bold.nii.gz"
		echo "${parent_dir}${subj_num}func/${subj_num}_${run_num}_task-rest_bold.nii.gz" >> "./dl_log.txt"

	done
done

