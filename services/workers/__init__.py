"""Worker scripts of the WonderZoom model services.

Each <svc>_worker.py runs as a separate process inside its own Python environment
(started by services/base.py:WorkerClient); they are not imported by the main process.
"""
