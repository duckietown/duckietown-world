#!/bin/zsh


for a in *.pdf; do
    convert $a -resize 2000 -shave 7x7  $a.jpg &
done
