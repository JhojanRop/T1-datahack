"""
Modelos Pydantic para endpoints del pipeline de clasificación
"""
from datetime import datetime

from pydantic import BaseModel, Field


class PipelineHealth(BaseModel):
    """Estado de salud del pipeline"""
    status: str = Field(..., description="Estado general: healthy, warning, error")
    biobert_status: str = Field(..., description="Estado del clasificador BioBERT")
    llm_status: str = Field(..., description="Estado del clasificador LLM")
    memory_usage_mb: float = Field(..., description="Uso de memoria en MB")
    cpu_usage_percent: float = Field(..., description="Uso de CPU en porcentaje")
    uptime_seconds: float = Field(..., description="Tiempo de actividad en segundos")
    last_check: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "biobert_status": "healthy",
                "llm_status": "healthy",
                "memory_usage_mb": 1250.5,
                "cpu_usage_percent": 15.2,
                "uptime_seconds": 86400,
                "last_check": "2025-08-26T10:30:00"
            }
        }


class PipelineStatistics(BaseModel):
    """Estadísticas del pipeline"""
    total_requests: int = Field(..., description="Total de requests procesados")
    successful_requests: int = Field(..., description="Requests exitosos")
    failed_requests: int = Field(..., description="Requests fallidos")
    avg_processing_time: float = Field(..., description="Tiempo promedio de procesamiento")
    biobert_usage_count: int = Field(..., description="Veces que se usó BioBERT")
    llm_usage_count: int = Field(..., description="Veces que se usó LLM")
    domains_detected: dict[str, int] = Field(..., description="Conteo por dominio detectado")
    hourly_stats: dict[str, int] = Field(..., description="Estadísticas por hora")
    error_breakdown: dict[str, int] = Field(..., description="Desglose de errores")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "total_requests": 5420,
                "successful_requests": 5380,
                "failed_requests": 40,
                "avg_processing_time": 0.85,
                "biobert_usage_count": 4100,
                "llm_usage_count": 1320,
                "domains_detected": {
                    "cardiovascular": 1250,
                    "neurological": 980,
                    "oncological": 850,
                    "hepatorenal": 720
                },
                "hourly_stats": {
                    "08:00": 120,
                    "09:00": 250,
                    "10:00": 180
                },
                "error_breakdown": {
                    "timeout": 15,
                    "memory_error": 5,
                    "invalid_input": 20
                }
            }
        }


class PipelineConfig(BaseModel):
    """Configuración del pipeline"""
    confidence_threshold: float = Field(..., description="Umbral de confianza global")
    llm_threshold: float = Field(..., description="Umbral para activar LLM")
    max_batch_size: int = Field(..., description="Tamaño máximo de batch")
    timeout_seconds: int = Field(..., description="Timeout en segundos")
    enable_caching: bool = Field(..., description="Habilitar caché")
    cache_ttl_minutes: int = Field(..., description="TTL del caché en minutos")
    log_level: str = Field(..., description="Nivel de logging")
    parallel_processing: bool = Field(..., description="Procesamiento en paralelo")

    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.7,
                "llm_threshold": 0.6,
                "max_batch_size": 50,
                "timeout_seconds": 30,
                "enable_caching": True,
                "cache_ttl_minutes": 60,
                "log_level": "INFO",
                "parallel_processing": True
            }
        }


class ConfigUpdate(BaseModel):
    """Request para actualizar configuración"""
    confidence_threshold: float | None = Field(None, description="Nuevo umbral de confianza", ge=0.0, le=1.0)
    llm_threshold: float | None = Field(None, description="Nuevo umbral para LLM", ge=0.0, le=1.0)
    max_batch_size: int | None = Field(None, description="Nuevo tamaño máximo de batch", ge=1, le=1000)
    timeout_seconds: int | None = Field(None, description="Nuevo timeout", ge=5, le=300)
    enable_caching: bool | None = Field(None, description="Habilitar/deshabilitar caché")
    cache_ttl_minutes: int | None = Field(None, description="Nuevo TTL del caché", ge=1, le=1440)
    log_level: str | None = Field(None, description="Nuevo nivel de logging")
    parallel_processing: bool | None = Field(None, description="Habilitar/deshabilitar procesamiento paralelo")

    class Config:
        schema_extra = {
            "example": {
                "confidence_threshold": 0.75,
                "llm_threshold": 0.65,
                "max_batch_size": 100,
                "log_level": "DEBUG"
            }
        }
