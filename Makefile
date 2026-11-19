.PHONY: install ingest test lint eval eval-fake api ui docker

install:
	pip install -e ".[dev,ui,rerank]"

ingest:          ## index the demo corpus (replace with your own folder)
	docqa ingest corpus --reset

test:
	LLM_PROVIDER=fake pytest -q

lint:
	ruff check src tests evals

eval:            ## real models, e.g. LLM_PROVIDER=ollama make eval
	python -m evals.run_eval

eval-fake:       ## offline: real retrieval ablation (hashing embeddings), meaningless answer scores
	LLM_PROVIDER=fake python -m evals.run_eval --no-judge

api:
	uvicorn verified_doc_qa.api:app --reload --port 8000

ui:
	streamlit run src/verified_doc_qa/ui.py

docker:
	docker compose up --build
