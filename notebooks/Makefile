
#targets=$(addprefix handle-,$(patsubst .ipynb,,$(wildcard *.ipynb)))

#all: $(targets)
#handle-%:
#	jupyter nbconvert --to html --execute "$*"
#	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$*"

targets_html=$(patsubst .ipynb,.html,$(wildcard *.ipynb))

all: $(targets_html)

clean:
	rm -f $(targets_html)
	 
%.html: %.ipynb
	jupyter nbconvert --to html --execute "$*"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$*"



#clean-output:
#	$(zsh for a in *.ipynb; jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $a)
