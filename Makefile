.PHONY: test lint eval dev frontend

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

eval:
	python eval/run_eval.py

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8741

frontend:
	npm --prefix frontend run dev
