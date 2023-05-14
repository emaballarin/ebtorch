.PHONY: clean
clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	rm -R -f ./.mypy_cache

.PHONY: format
format:
	find . -type f -name '*.py' -exec reorder-python-imports --py310-plus "{}" \;
	black "$(realpath .)"
