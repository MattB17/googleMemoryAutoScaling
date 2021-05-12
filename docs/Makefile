# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../../memory_auto_scaling_docs
PDFBUILDDIR   = source
PDF           = ../manual.pdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PDFBUILDDIR) $(PDF)
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR) all-pdf
	cp $(PDFBUILDDIR)*.pdf $(PDF)
	@eco "pdflatex finished; see $(PDF)"

buildandcommit: html
	cd $(BUILDDIR)/html; git add -A; git commit -m "Rebuilt docs"; git push origin gh-pages
