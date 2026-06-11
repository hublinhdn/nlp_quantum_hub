# Build AJDSAI/AJEB submission (Emerald house style)
# Usage:
#   make             — build PDF
#   make docx        — convert to Word (.docx) matching the PDF style
#   make wc          — count words (target ~6,000–8,000)
#   make clean       — remove aux files

TEX = main
LATEX = pdflatex
BIB = bibtex
PY = ../.venv_local/bin/python3

.PHONY: all build pdf-only quick clean wc docx

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

# Convert to Word, matching the PDF style (numbered captions, booktabs
# tables, centred figures, OMML equations, Emerald-Harvard references).
# Requires: pandoc, the venv python with python-docx, harvard-emerald.csl,
# and the TikZ pipeline rendered to figures/00_pipeline.png.
docx: $(TEX).tex references.bib harvard-emerald.csl docx_style.py figures/00_pipeline.png
	$(PY) -c "import re;src=open('$(TEX).tex').read();open('_docx_src.tex','w').write(re.sub(r'\\\\begin\{tikzpicture\}.*?\\\\end\{tikzpicture\}', r'\\\\includegraphics[width=0.65\\\\linewidth]{figures/00_pipeline.png}', src, flags=re.DOTALL))"
	pandoc _docx_src.tex --from=latex --to=docx --bibliography=references.bib --citeproc --csl=harvard-emerald.csl -o $(TEX).docx
	$(PY) docx_style.py $(TEX).docx
	@rm -f _docx_src.tex
	@echo "✓ Built $(TEX).docx"

# Render the TikZ pipeline figure to PNG for the Word version
figures/00_pipeline.png: main.tex
	$(PY) -c "import re;src=open('main.tex').read();m=re.search(r'\\\\begin\{tikzpicture\}.*?\\\\end\{tikzpicture\}',src,re.DOTALL);open('_pipe.tex','w').write(r'\documentclass[tikz,border=10pt]{standalone}'+chr(10)+r'\usepackage{tikz}'+chr(10)+r'\usetikzlibrary{positioning,arrows.meta}'+chr(10)+r'\begin{document}'+chr(10)+m.group(0)+chr(10)+r'\end{document}'+chr(10))"
	-$(LATEX) -interaction=nonstopmode _pipe.tex
	sips -s format png --resampleHeight 1200 _pipe.pdf --out figures/00_pipeline.png
	@rm -f _pipe.tex _pipe.pdf _pipe.aux _pipe.log

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
