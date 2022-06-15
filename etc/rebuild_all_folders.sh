#!/bin/bash

#cd /data/jsreyl/TEMNet/TEMNet/multiviral_dataset/context_virus_RAW/validation
for virdir in `ls`;
do
	echo "Processing ${virdir} from {`pwd`}"
	cd ${virdir}
	bash ../../rebuild_folder_Structure.sh ${virdir,,} #Use lowercase of folder to uniquely identify images
	cd ..
done
