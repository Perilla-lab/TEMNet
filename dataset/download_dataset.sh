#!bin/bash

#Create list of file ids and file names for downloading from google-drive
file_ids=("1qz7WBKaxwPm2MPn5e98Zyp3qw5dH3drh" "1p63sWu0YDUXzGC7fVtF3EsSYcrfYAQAQ")
file_names=("backbone_dataset.zip"  "rcnn_dataset_full.zip")

# download each file via wget
for i in ${!file_ids[@]};
do
    file_id=${file_ids[$i]}
    file_name=${file_names[$i]}
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file_id} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O ${file_name} && rm -rf /tmp/cookies.txt
done
