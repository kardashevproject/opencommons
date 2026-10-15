.PHONY: test paper clean

test:
	PYTHONPATH=src pytest -q

paper:
	XDG_CACHE_HOME=$(CURDIR)/.cache tectonic --only-cached paper/main.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
