"""Prints which of the modules named on the command line this Python cannot import.

Used by the launcher scripts on both platforms to test candidate interpreters. It is a
file rather than a -c one-liner because quoting a python snippet inside a Windows batch
FOR loop is a reliable source of silent breakage.

find_spec, not import: importing totalsegmentator pulls in torch and takes seconds,
which is too slow to repeat across every environment on a machine. Whichever
interpreter wins the search is import-tested for real afterwards.
"""
import importlib.util
import sys

missing = []
for name in sys.argv[1:]:
    try:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    except (ImportError, ValueError):
        # A package whose parent is broken or half-removed reads as missing.
        missing.append(name)

print(" ".join(missing))
