.PHONY: lint test

lint:
	python -m ruff check .

test:
	python -m pytest tests/ -v
