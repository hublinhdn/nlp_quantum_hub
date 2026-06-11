# Build AJDSAI/AJEB submission (Emerald house style)
# Usage:
#   make             — build PDF
#   make wc          — count words (target ~6,000–8,000)
#   make clean       — remove aux files

TEX = main
LATEX = pdflatex
BIB = bibtex

.PHONY: all build pdf-only quick clean wc

all: pdf-only

pdf-only: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(BIB) $(TEX)
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	@echo ""
	@echo "✓ Built $(TEX).pdf"
	@which pdfinfo > /dev/null 2>&1 && pdfinfo $(TEX).pdf | grep -E "Pages|File size" || true

quick:
	$(LATEX) -interaction=nonstopmode $(TEX).tex

# Word count (using detex if available, fallback to wc)
wc:
	@if command -v detex >/dev/null 2>&1; then \
	  echo "Word count (detex + wc):"; \
	  detex $(TEX).tex | wc -w; \
	else \
	  echo "Word count (rough, raw tex):"; \
	  wc -w $(TEX).tex; \
	fi

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc $(TEX).bbl $(TEX).blg
	rm -f $(TEX).fdb_latexmk $(TEX).fls $(TEX).synctex.gz
