.PHONY: dev test lint check gen-api build-ui app recache

dev:  ## api + ui dev servers together, Ctrl+C stops both
	@trap 'kill 0' INT TERM EXIT; \
	uv run python -m fovea.api.app & \
	( cd ui && pnpm dev ) & \
	wait

recache:  ## clear pipeline cache, keep user verdicts: make recache DIR=<folder>
	@test -n "$(DIR)" || { echo "usage: make recache DIR=<folder>"; exit 1; }
	@test -f "$(DIR)/.fovea/cache.sqlite" || { echo "no cache at $(DIR)/.fovea"; exit 1; }
	@sqlite3 "$(DIR)/.fovea/cache.sqlite" "delete from entries where kind != 'user'"
	@echo "cleared pipeline cache in $(DIR), reopen the folder to recompute"

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
	cd ui && pnpm exec tsc -b && pnpm lint

check: lint test

gen-api:  ## regenerate ui/src/lib/api-types.d.ts from the FastAPI schema
	uv run python -c "from fovea.api.app import create_app; import json; \
	print(json.dumps(create_app().openapi()))" > ui/openapi.json
	cd ui && pnpm exec openapi-typescript openapi.json -o src/lib/api-types.d.ts && rm openapi.json

build-ui:
	cd ui && pnpm build

app: build-ui  ## fresh ui build + frozen dist/fovea.app
	uv run pyinstaller packaging/fovea.spec --noconfirm --distpath dist --workpath build
	@echo "done: open dist/fovea.app"
