.PHONY: setup verify prices extract graph results figures site paper test privacy all serve
UV ?= uv

setup:
	$(UV) sync

verify:
	$(UV) run python scripts/verify_tickers.py

prices:
	$(UV) run python scripts/download_prices.py

extract:
	$(UV) run python scripts/run_extract.py

graph:
	$(UV) run python scripts/build_graph_json.py

results:
	$(UV) run python scripts/run_all.py

figures site paper:
	@echo "target '$@' is built in the second stage of the project"

test:
	$(UV) run pytest -q

privacy:
	$(UV) run python -m scgraph.privacy

all: setup verify prices extract graph results test privacy

serve:
	cd site/dist && python3 -m http.server 8000
