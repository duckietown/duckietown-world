#!/bin/zsh

for a in *.html; do
    wkhtmltopdf "$a" "$a.pdf"
done
