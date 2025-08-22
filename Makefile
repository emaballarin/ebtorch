RUFF_CONFIG := ~/ruffconfigs/ebdefault/ruff.toml
PYTHON_FILES := $(shell find . -name "*.py" -type f)
REQUIREMENTS_FILES := $(shell find . -name "requirements.txt" -type f)

.PHONY: help clean format deployhooks precau precra gitall check-git-status

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	@find . -type d \( -name ".mypy_cache" -o -name "__pycache__" -o -name ".ruff_cache" \) -exec rm -rf {} + 2>/dev/null || true

format:
	@if [ -n "$(PYTHON_FILES)" ]; then \
		for file in $(PYTHON_FILES); do \
			reorder-python-imports --py310-plus "$$file" || exit 1; \
		done; \
		ruff format --config "$(RUFF_CONFIG)" . || exit 1; \
	fi
	@if [ -n "$(REQUIREMENTS_FILES)" ]; then \
		for file in $(REQUIREMENTS_FILES); do \
			sort-requirements "$$file" || exit 1; \
		done; \
	fi

deployhooks:
	@if [ -d ./.githooks ]; then \
		cp -f ./.githooks/* ./.git/hooks/ && \
		chmod +x ./.git/hooks/* && \
	else \
		exit 1; \
	fi

precau:
	@pre-commit autoupdate

precra:
	@pre-commit run --all-files

check-git-status:
	@if [ -z "$$(git status --porcelain)" ]; then \
		exit 1; \
	fi

gitall: check-git-status
	@git add -A
	@git commit --all
	@git push


lint: precra
fmt: format

gitpre: format precau precra clean
gitpush: format precau precra clean gitall
clfmt: format clean
