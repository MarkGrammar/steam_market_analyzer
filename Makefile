.PHONY: app_list details normalize duckdb clean_basic all

app_list:
	python src/ingest/fetch_app_list.py

details:
	python -m src.ingest.fetch_details

normalize:
	python -m src.ingest.normalize_details

duckdb:
	python -m src.ingest.to_duckdb

clean_basic:
	python -m src.clean.clean_basics
	
tfidf:
	python -m src.features.build_tfidf
	

all: app_list details normalize duckdb clean_basic


.PHONY: search_demo
search_demo:
	python -c "from src.search.hybrid import search_hybrid_text; import pandas as pd; \
print(search_hybrid_text('third person action rpg, 3D, co-op', k_return=10).head())"


.PHONY: embed similar

embed:
	python -m src.features.build_embeddings

# Örnek kullanım:
# make similar TEXT="cozy farming pixel art"
# make similar APP_ID=1086940
similar:
ifdef TEXT
	python -m src.features.query_similar --text "$(TEXT)" --topk 5
else ifdef APP_ID
	python -m src.features.query_similar --app_id $(APP_ID) --topk 5
else
	@echo 'Provide TEXT="..." or APP_ID=12345'
	@exit 1
endif
