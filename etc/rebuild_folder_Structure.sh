#!/bin/bash/

for imgname in `ls *.tif`;
do 
	dirname="${imgname%.*}"
	mkdir -p restructured/${dirname}_${1}
	cp ${imgname} restructured/${dirname}_${1}/${dirname}_${1}.tif
	cp csvs_reformat/region_data_${dirname}.csv restructured/${dirname}_${1}/region_data_${dirname}_${1}.csv
done 
