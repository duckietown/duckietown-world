
#targets=$(addprefix handle-,$(patsubst .ipynb,,$(wildcard *.ipynb)))

#all: $(targets)
#handle-%:
#	jupyter nbconvert --to html --execute "$*"
#	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$*"

targets_html=$(patsubst %.ipynb,%.html,$(wildcard *.ipynb))

all: $(targets_html)

clean: cleanup
	rm -f $(targets_html)

%.html: %.ipynb
	jupyter nbconvert --to html --execute "$*"

#%.pdf: %.ipynb
#	jupyter nbconvert --to html --execute "$*"

fatten:
	jupyter nbconvert --inplace --execute *.ipynb

cleanup:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb



#clean-output:
#	$(zsh for a in *.ipynb; jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $a)
