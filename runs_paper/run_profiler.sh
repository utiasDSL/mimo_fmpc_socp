#!/bin/bash
# Wrapper script to run kernprof with the correct Python environment
python -m kernprof -l -v "$@"
