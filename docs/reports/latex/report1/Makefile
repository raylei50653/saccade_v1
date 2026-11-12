.PHONY: all clean
all: report.pdf

report.pdf: report.tex numeric-tables.tex schedule-results.tex latency-percentiles.tex assets/story-pipeline.pdf assets/story-example.pdf assets/story-schedule.pdf assets/story-datasets.pdf assets/story-group-deltas.pdf assets/story-speed.pdf
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex

clean:
	latexmk -c report.tex
