# Bifolia-ordering report. `make` builds the PDF; `make figures` regenerates figures/*.pdf.
TEX = bifolia_order
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all figures clean distclean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(wildcard figures/*.pdf)
	$(LATEXMK) $(TEX).tex

figures:
	cd figures && uv run python make_all.py

clean:
	latexmk -C $(TEX).tex

distclean: clean
	rm -f figures/fig_*.pdf figures/fig_*.png
