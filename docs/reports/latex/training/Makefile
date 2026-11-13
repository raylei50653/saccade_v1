.PHONY: all clean
all: training.pdf

training.pdf: training.tex milestone-table.tex assets/train-chain.pdf assets/train-data.pdf assets/train-curves.pdf assets/train-progress.pdf assets/train-deploy.pdf
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error training.tex

clean:
	latexmk -c training.tex
