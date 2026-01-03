# UV Commands Cheat Sheet

## Add packages from requirements.txt

uv add --requirements requirements.txt

## Install dependencies from uv.lock

uv sync --frozen

## Update lockfile after changing pyproject.toml

uv lock

## Run the scripts

uv run src/\*.py

## Run the app (streamlit)

streamlit run src/app.py
