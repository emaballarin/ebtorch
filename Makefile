.PHONY: clean
clean:
	rm -R -f ./.mypy_cache
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete -o -type d -name .mypy_cache -delete

.PHONY: format
format:
	find . -type f -name '*.py' -exec reorder-python-imports --py310-plus "{}" \;
	black "$(realpath .)"
	sort-requirements "$(realpath .)/requirements.txt"

.PHONY: deployhooks
deployhooks:
	cp -f ./.githooks/* ./.git/hooks/

.PHONY: precau
precau:
	pre-commit autoupdate

.PHONY: precra
precra:
	pre-commit run --all-files

.PHONY: gitall
gitall:
	git add -A; git commit --all; git push
