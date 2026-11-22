.PHONY: setup verify prices extract graph results figures shock stats site paper readme hero test privacy all serve
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

stats:
	$(UV) run python scripts/run_node_stats.py

shock:
	$(UV) run python scripts/run_shock_case.py

figures:
	$(UV) run python scripts/make_figures.py

site:
	$(UV) run python site/build.py

paper:
	$(UV) run python scripts/build_paper.py

readme:
	$(UV) run python scripts/build_readme.py

hero:
	$(UV) run python scripts/record_hero.py

test:
	$(UV) run pytest -q

privacy:
	$(UV) run python -m scgraph.privacy

all: setup verify prices extract graph results figures shock stats paper site readme test privacy

serve:
	cd site/dist && python3 -m http.server 8000
