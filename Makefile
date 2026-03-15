PYTHON = /opt/anaconda3/bin/python3

.PHONY: setup db up down ingest run clean

setup:
	pip install -r requirements.txt

db:
	docker compose up -d
	@echo "PostgreSQL ready at localhost:5433"

up: db
	@sleep 3
	PYTHONPATH=. $(PYTHON) src/ingestion/fetch_data.py
	@echo "Data ingested. Starting dashboard..."
	streamlit run app.py

ingest:
	PYTHONPATH=. $(PYTHON) src/ingestion/fetch_data.py

run:
	streamlit run app.py

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f outputs/*.csv outputs/*.json
