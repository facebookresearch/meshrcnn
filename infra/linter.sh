#!/bin/bash -ev
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Run this script at project root by "./infra/linter.sh" before you commit.

echo "Running isort..."
isort -y --multi-line 3 --trailing-comma -sp . --skip datasets --skip-glob '*/__init__.py' --atomic

echo "Running black..."
black -l 100 .

echo "Running flake8..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
