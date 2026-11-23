#!/bin/sh
# A stable executable avoids per-test script startup latency on macOS.
if [ "$1" = "--ready" ]; then
    exit 0
fi
{
    IFS= read -r python
    IFS= read -r script
    IFS= read -r control
    IFS= read -r log
} < "$0.args"
exec "$python" -I -S "$script" "$control" "$log" "$0" "$@"
