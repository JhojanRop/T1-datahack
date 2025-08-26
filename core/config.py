"""
Configuración central de la aplicación
"""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración principal de la aplicación"""

    # Información de la aplicación
    app_name: str = "Medical Classification API"
    app_version: str = "1.0.0"
    app_description: str = "API para clasificación de literatura médica usando BioBERT y LLM"

    # Configuración del servidor
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Configuración de base de datos
    database_url: str = "sqlite:///./medical_classification.db"

    # Configuración de autenticación
    secret_key: str = "medical-classification-api-secret-key-2025"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Configuración de modelos
    biobert_model_path: str = "./model/biobert_finetuned_v3"
    confidence_threshold: float = 0.7
    llm_threshold: float = 0.6

    # Configuración de LLM (Gemini)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Configuración de pipeline
    max_batch_size: int = 50
    timeout_seconds: int = 30
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    parallel_processing: bool = True

    # Configuración de logging
    log_level: str = "INFO"
    log_file: str = "logs/medical_api.log"
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5

    # Configuración de monitoreo
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Configuración de CORS
    allowed_origins: list[str] = ["*"]
    allowed_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers: list[str] = ["*"]

    # Configuración de rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # segundos

    # Paths
    data_dir: str = "./data"
    model_dir: str = "./model"
    logs_dir: str = "./logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Configuración para desarrollo"""
    debug: bool = True
    log_level: str = "DEBUG"
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]


class ProductionSettings(Settings):
    """Configuración para producción"""
    debug: bool = False
    log_level: str = "WARNING"
    allowed_origins: list[str] = []  # Debe ser configurado específicamente


class TestingSettings(Settings):
    """Configuración para testing"""
    database_url: str = "sqlite:///./test_medical_classification.db"
    access_token_expire_minutes: int = 5
    enable_caching: bool = False


def get_settings() -> Settings:
    """Obtener configuración basada en el entorno"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Instancia global de configuración
settings = get_settings()
