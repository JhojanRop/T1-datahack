"""
Modelos Pydantic para endpoints administrativos
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """Estado general del sistema"""
    api_status: str = Field(..., description="Estado de la API")
    database_status: str = Field(..., description="Estado de la base de datos")
    model_status: str = Field(..., description="Estado de los modelos")
    memory_usage: dict[str, float] = Field(..., description="Uso de memoria por componente")
    disk_usage: dict[str, float] = Field(..., description="Uso de disco")
    active_connections: int = Field(..., description="Conexiones activas")
    system_load: float = Field(..., description="Carga del sistema")
    uptime: float = Field(..., description="Tiempo de actividad en segundos")
    version: str = Field(..., description="Versión del sistema")
    last_check: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "api_status": "healthy",
                "database_status": "healthy",
                "model_status": "healthy",
                "memory_usage": {
                    "api": 512.0,
                    "biobert": 1024.0,
                    "llm": 256.0
                },
                "disk_usage": {
                    "total_gb": 100.0,
                    "used_gb": 45.2,
                    "available_gb": 54.8
                },
                "active_connections": 25,
                "system_load": 0.75,
                "uptime": 86400,
                "version": "1.0.0"
            }
        }


class MaintenanceResult(BaseModel):
    """Resultado de operaciones de mantenimiento"""
    operation_type: str = Field(..., description="Tipo de operación de mantenimiento")
    success: bool = Field(..., description="Si la operación fue exitosa")
    details: dict[str, Any] = Field(..., description="Detalles de la operación")
    duration_seconds: float = Field(..., description="Duración en segundos")
    errors: list[str] = Field(default_factory=list, description="Errores encontrados")
    warnings: list[str] = Field(default_factory=list, description="Advertencias")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "operation_type": "cache_cleanup",
                "success": True,
                "details": {
                    "cache_entries_removed": 1250,
                    "memory_freed_mb": 128.5,
                    "files_cleaned": 45
                },
                "duration_seconds": 5.2,
                "errors": [],
                "warnings": ["Cache was 85% full"]
            }
        }


class OptimizationResult(BaseModel):
    """Resultado de optimización del sistema"""
    optimization_type: str = Field(..., description="Tipo de optimización realizada")
    before_metrics: dict[str, float] = Field(..., description="Métricas antes de la optimización")
    after_metrics: dict[str, float] = Field(..., description="Métricas después de la optimización")
    improvement_percentage: dict[str, float] = Field(..., description="Porcentaje de mejora por métrica")
    recommendations: list[str] = Field(..., description="Recomendaciones adicionales")
    duration_seconds: float = Field(..., description="Duración de la optimización")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "optimization_type": "model_quantization",
                "before_metrics": {
                    "avg_inference_time": 1.2,
                    "memory_usage_mb": 1024.0,
                    "accuracy": 0.87
                },
                "after_metrics": {
                    "avg_inference_time": 0.8,
                    "memory_usage_mb": 512.0,
                    "accuracy": 0.86
                },
                "improvement_percentage": {
                    "inference_time": 33.3,
                    "memory_usage": 50.0,
                    "accuracy": -1.1
                },
                "recommendations": [
                    "Monitor accuracy closely",
                    "Consider batch processing for better throughput"
                ],
                "duration_seconds": 120.5
            }
        }


class LogEntry(BaseModel):
    """Entrada de log del sistema"""
    timestamp: datetime = Field(..., description="Timestamp del log")
    level: str = Field(..., description="Nivel del log: DEBUG, INFO, WARNING, ERROR")
    component: str = Field(..., description="Componente que generó el log")
    message: str = Field(..., description="Mensaje del log")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2025-08-26T10:30:15.123Z",
                "level": "INFO",
                "component": "biobert_classifier",
                "message": "Article classified successfully",
                "metadata": {
                    "article_id": "art_12345",
                    "processing_time": 0.85,
                    "confidence": 0.92
                }
            }
        }
