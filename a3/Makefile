SLIDES=slides
BEAMER_JUNK=$(SLIDES).nav $(SLIDES).toc $(SLIDES).out $(SLIDES).snm $(SLIDES).toc $(SLIDES).vrb
.PHONY: all clean purge

quick:
	pdflatex $(SLIDES)

slides:
	pdflatex ${SLIDES}
	bibtex ${SLIDES}
	pdflatex ${SLIDES}
	pdflatex ${SLIDES}

clean:
	rm -f *.aux *.blg *.bbl *.log $(BEAMER_JUNK)

purge:  clean
	rm -f $(SLIDES).pdf
