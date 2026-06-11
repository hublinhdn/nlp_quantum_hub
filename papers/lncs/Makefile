# Build the LaTeX report (Springer LNCS format)
# Usage:
#   make             — sync figures + build PDF
#   make pdf-only    — chỉ build PDF (giả sử figures đã sync)
#   make sync-figures — sync PNG từ ../results/ → ./figures/
#   make clean       — remove aux files
#   make watch       — rebuild on change (needs latexmk)

TEX = main
LATEX = pdflatex
BIB = bibtex

.PHONY: all build pdf-only clean watch sync-figures distclean

all: sync-figures pdf-only

# Full build (chạy pdflatex + bibtex 4 lần để cross-ref + bib hoàn tất)
pdf-only: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(BIB) $(TEX)
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	@echo ""
	@echo "✓ Built $(TEX).pdf"
	@which pdfinfo > /dev/null 2>&1 && pdfinfo $(TEX).pdf | grep -E "Pages|File size" || true

# Quick rebuild (no bib refresh — faster for content tweaks)
quick:
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	@echo "✓ Quick rebuild done"

# Auto-rebuild on file change (latexmk)
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TEX).tex

# Clean aux files (giữ PDF)
clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc $(TEX).bbl $(TEX).blg
	rm -f $(TEX).fdb_latexmk $(TEX).fls $(TEX).synctex.gz
	rm -f *.aux

# Distclean: also remove the PDF
distclean: clean
	rm -f $(TEX).pdf
	rm -rf figures

# Sync figures từ ../results/ vào ./figures/
sync-figures:
	@mkdir -p figures
	@echo "[sync] Result plots ..."
	@if ls ../results/final/plots/*.png 1> /dev/null 2>&1; then \
	  cp -v ../results/final/plots/*.png figures/ | sed 's/^/  /'; \
	else \
	  echo "  [WARN] Không có ../results/final/plots/*.png"; \
	  echo "  → Chạy: python scripts/08_final_analysis.py"; \
	fi
	@echo ""
	@echo "[sync] Spider diagram example (prefer short-word phrases) ..."
	@first=""; \
	for cand in pure_comedy very_cute really_slick otherwise_good a_brilliant; do \
	  match=$$(ls ../results/diagrams/spiders/train/spiders_train_*POS_$${cand}.png 2>/dev/null | head -1); \
	  if [ -n "$$match" ]; then first="$$match"; break; fi; \
	done; \
	if [ -z "$$first" ]; then \
	  first=$$(ls ../results/diagrams/spiders/train/spiders_train_*POS*.png 2>/dev/null | head -1); \
	fi; \
	if [ -n "$$first" ]; then \
	  cp -v "$$first" figures/spiders_example.png | sed 's/^/  /'; \
	else \
	  echo "  [WARN] Không có spider diagram"; \
	fi
	@echo ""
	@echo "[sync] Cups diagram example (prefer short-word phrases) ..."
	@first=""; \
	for cand in pure_comedy very_cute really_slick otherwise_good a_brilliant; do \
	  match=$$(ls ../results/diagrams/cups/train/cups_train_*POS_$${cand}.png 2>/dev/null | head -1); \
	  if [ -n "$$match" ]; then first="$$match"; break; fi; \
	done; \
	if [ -z "$$first" ]; then \
	  first=$$(ls ../results/diagrams/cups/train/cups_train_*POS*.png 2>/dev/null | head -1); \
	fi; \
	if [ -n "$$first" ]; then \
	  cp -v "$$first" figures/cups_example.png | sed 's/^/  /'; \
	else \
	  echo "  [WARN] Không có cups diagram"; \
	fi
	@echo ""
	@echo "[sync] Done. Files in figures/:"
	@ls -1 figures/ 2>/dev/null | sed 's/^/  /'
