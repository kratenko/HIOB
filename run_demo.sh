#!/bin/sh

# make sure we have the demo sample:
if [ ! -f demo/data/tb100/Deer.zip ]; then
	mkdir -p demo/data/tb100
	cd demo/data/tb100
	../../../bin/hiob_downloader deer
	cd ../../..
else
	echo "Demo sample seems to be at place."
	echo "If it is broken, try deleting directory 'demo/data'."
fi

# run the tracking
python hiob_gui.py -e demo/environment.yaml -t demo/tracker.yaml

