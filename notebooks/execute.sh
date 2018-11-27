#!/bin/bash

for a in *.ipynb; do

    jupyter nbconvert --to html --execute "$a"
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$a"

done
