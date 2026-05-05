PAPER := main

.PHONY: all watch clean veryclean view

all:
	latexmk $(PAPER).tex

watch:
	latexmk -pvc $(PAPER).tex

clean:
	latexmk -c

veryclean:
	latexmk -C

view: all
	@xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf 2>/dev/null || echo "Open $(PAPER).pdf manually"
