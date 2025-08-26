"""
Sistema de logging para la API médica
"""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings


class CustomFormatter(logging.Formatter):
    """Formateador personalizado para logs con colores en consola"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self.FORMATS = {
            logging.DEBUG: self.grey + self._get_format() + self.reset,
            logging.INFO: self.blue + self._get_format() + self.reset,
            logging.WARNING: self.yellow + self._get_format() + self.reset,
            logging.ERROR: self.red + self._get_format() + self.reset,
            logging.CRITICAL: self.bold_red + self._get_format() + self.reset
        }

    def _get_format(self):
        return "[%(asctime)s] %(name)s | %(levelname)s | %(message)s"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class MedicalAPILogger:
    """Logger centralizado para la API médica"""

    def __init__(self):
        self.logger = logging.getLogger("medical_api")
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))

        # Evitar duplicar handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Configurar handlers de logging"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(console_handler)

        # File handler
        if settings.log_file:
            self._ensure_log_directory()
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _ensure_log_directory(self):
        """Asegurar que el directorio de logs existe"""
        log_dir = Path(settings.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str = None) -> logging.Logger:
        """Obtener logger específico"""
        if name:
            return logging.getLogger(f"medical_api.{name}")
        return self.logger


# Instancia global del logger
api_logger = MedicalAPILogger()
logger = api_logger.get_logger()


def get_logger(name: str = None) -> logging.Logger:
    """Función helper para obtener logger"""
    return api_logger.get_logger(name)


def log_request(request_id: str, method: str, path: str, user: str = None):
    """Log de request entrante"""
    user_info = f" | User: {user}" if user else ""
    logger.info(f"REQUEST {request_id} | {method} {path}{user_info}")


def log_response(request_id: str, status_code: int, duration: float):
    """Log de response"""
    logger.info(f"RESPONSE {request_id} | Status: {status_code} | Duration: {duration:.3f}s")


def log_error(request_id: str, error: Exception, context: dict[str, Any] = None):
    """Log de error con contexto"""
    context_str = f" | Context: {context}" if context else ""
    logger.error(f"ERROR {request_id} | {type(error).__name__}: {error}{context_str}")


def log_classification(request_id: str, method: str, domains: list[str], confidence: float, duration: float):
    """Log específico para clasificaciones"""
    classification_logger = get_logger("classification")
    classification_logger.info(
        f"CLASSIFICATION {request_id} | Method: {method} | Domains: {domains} | "
        f"Confidence: {confidence:.3f} | Duration: {duration:.3f}s"
    )


def log_system_metric(metric_name: str, value: Any, component: str = "system"):
    """Log de métricas del sistema"""
    metrics_logger = get_logger("metrics")
    metrics_logger.info(f"METRIC | {component}.{metric_name}: {value}")


def log_security_event(event_type: str, details: dict[str, Any], severity: str = "WARNING"):
    """Log de eventos de seguridad"""
    security_logger = get_logger("security")
    level = getattr(logging, severity.upper())
    security_logger.log(level, f"SECURITY | {event_type} | {details}")


class RequestLogger:
    """Context manager para logging de requests"""

    def __init__(self, request_id: str, method: str, path: str, user: str = None):
        self.request_id = request_id
        self.method = method
        self.path = path
        self.user = user
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        log_request(self.request_id, self.method, self.path, self.user)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type:
            log_error(self.request_id, exc_val)
            log_response(self.request_id, 500, duration)
        else:
            log_response(self.request_id, 200, duration)
