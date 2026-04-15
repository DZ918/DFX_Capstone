#!/bin/bash
export LD_LIBRARY_PATH="/home/user/.local/lib/python3.10/site-packages/cusparselt/lib:/home/user/.local/lib:${LD_LIBRARY_PATH:-}"
exec python3 main.py "$@"
