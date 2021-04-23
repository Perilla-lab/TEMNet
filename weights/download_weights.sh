#!bin/bash

#Create list of file ids and file names for downloading from google-drive
file_ids=("1zqE5kp3n2xYhGbn_gEd7y_iR3K-FDx5s" "151mE8Q3Or-rQSDupqCbpKPr-i6Q15Fh3" "19ijtoiNb4Z0W7dM1V7FwctPfR8exiUwQ")
file_names=("rcnn_temnet_weights_gn_res512.h5"    "rcnn_resnet101_weights_res512.h5"    "rcnn_resnet101v2_weights_res512.h5")

# download each file via wget
for i in ${!file_ids[@]};
do
    file_id=${file_ids[$i]}
    file_name=${file_names[$i]}
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file_id} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O ${file_name} && rm -rf /tmp/cookies.txt
done
