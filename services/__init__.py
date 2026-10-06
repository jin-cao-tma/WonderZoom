"""WonderZoom model services.

Gen3C, Chain-of-Zoom and Step1X-Edit each run as a persistent worker process inside their own
Python environment and are driven over newline-delimited JSON (services/workers/_wz_protocol.py).
This package only needs the standard library and omegaconf, so it imports in any environment.

    from services import ServiceManager, ServiceError, load_services_config
"""
from .base import ServiceError, WorkerClient
from .config import SERVICE_NAMES, load_services_config
from .gpu_arbiter import MAIN_TENANT, GpuArbiter
from .manager import ServiceManager

__all__ = ["ServiceError", "WorkerClient", "ServiceManager", "GpuArbiter", "load_services_config",
           "SERVICE_NAMES", "MAIN_TENANT"]
