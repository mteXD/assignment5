FILENAME = report
BIBER = biber # change this to `BIBER = biber` if using a bibliography

all: # clean
	@lualatex --shell-escape $(FILENAME).tex
	@clear
	@$(BIBER) $(FILENAME)
	@clear
	@lualatex --shell-escape $(FILENAME).tex
	@clear
	@lualatex --shell-escape $(FILENAME).tex

clean:
	$(RM) *.log *.run.xml *.bcf *.blg *.out *.bbl *.aux *.pdf
