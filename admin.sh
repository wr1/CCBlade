#!/bin/bash
# Admin script for CCBlade refactoring

# List of files added/modified
src/ccblade/__init__.py
src/ccblade/core/__init__.py
src/ccblade/core/bem.py
src/ccblade/core/ccblade.py
src/ccblade/airfoil/__init__.py
src/ccblade/airfoil/airfoil.py
src/ccblade/utils/__init__.py
src/ccblade/utils/csystem.py
src/ccblade/components/__init__.py
src/ccblade/components/ccblade_component.py
src/ccblade/cli/__init__.py
src/ccblade/cli/main.py
pyproject.toml
tests/test_airfoilprep.py
tests/test_ccblade.py

# Run ruff format
ruff format

# Run ruff check and fix
ruff check --fix > out.txt

# Run pytest and append to out.txt
uv run pytest -v --cov=ccblade >> out.txt

# Git commit each file
for file in \
    src/ccblade/__init__.py \
    src/ccblade/core/__init__.py \
    src/ccblade/core/bem.py \
    src/ccblade/core/ccblade.py \
    src/ccblade/airfoil/__init__.py \
    src/ccblade/airfoil/airfoil.py \
    src/ccblade/utils/__init__.py \
    src/ccblade/utils/csystem.py \
    src/ccblade/components/__init__.py \
    src/ccblade/components/ccblade_component.py \
    src/ccblade/cli/__init__.py \
    src/ccblade/cli/main.py \
    pyproject.toml \
    tests/test_airfoilprep.py \
    tests/test_ccblade.py; do
    if [ -f "$file" ]; then
        git add "$file"
        git commit -m "Refactor CCBlade: reorganize into subdirectories and update CLI options for $file"
    fi
done
