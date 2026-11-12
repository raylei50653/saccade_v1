.PHONY: all clean
all: dataflow.pdf

dataflow.pdf: dataflow.tex output-notes.tex assets/flow.pdf assets/state.pdf assets/leases.pdf assets/schedule.pdf
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error dataflow.tex

clean:
	latexmk -c dataflow.tex
