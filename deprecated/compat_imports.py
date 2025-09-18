"""Compatibility layer: if legacy code expected modules at project.<name>, ensure they resolve.
In most cases, updating PYTHONPATH to include project/src and using 'from project.src import <module>' is preferred."""
