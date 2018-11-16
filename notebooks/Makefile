
targets=$(addprefix handle-,$(patsubst .ipynb,,$(wildcard *.ipynb)))

all: $(targets)


handle-%:
	jupyter nbconvert --to html --execute "$*"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$*"


#clean-output:
#	$(zsh for a in *.ipynb; jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $a)
