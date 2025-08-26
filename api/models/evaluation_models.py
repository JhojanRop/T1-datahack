"""
Modelos Pydantic para endpoints de evaluación y métricas
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Request para evaluación de dataset"""
    dataset_path: str = Field(..., description="Ruta al dataset CSV para evaluar")
    method: str = Field("hybrid", description="Método de clasificación a evaluar")
    confidence_threshold: float = Field(0.7, description="Umbral de confianza", ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "dataset_path": "data/test_dataset.csv",
                "method": "hybrid",
                "confidence_threshold": 0.7
            }
        }


class EvaluationResult(BaseModel):
    """Resultado de evaluación"""
    accuracy: float = Field(..., description="Precisión general")
    precision_macro: float = Field(..., description="Precisión macro")
    recall_macro: float = Field(..., description="Recall macro")
    f1_macro: float = Field(..., description="F1-score macro")
    precision_micro: float = Field(..., description="Precisión micro")
    recall_micro: float = Field(..., description="Recall micro")
    f1_micro: float = Field(..., description="F1-score micro")
    precision_weighted: float = Field(..., description="Precisión ponderada")
    recall_weighted: float = Field(..., description="Recall ponderado")
    f1_weighted: float = Field(..., description="F1-score ponderado")
    confusion_matrix: dict[str, Any] = Field(..., description="Matriz de confusión")
    classification_report: dict[str, Any] = Field(..., description="Reporte detallado")
    processing_time: float = Field(..., description="Tiempo de evaluación")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.87,
                "precision_macro": 0.85,
                "recall_macro": 0.82,
                "f1_macro": 0.83,
                "precision_micro": 0.87,
                "recall_micro": 0.87,
                "f1_micro": 0.87,
                "precision_weighted": 0.86,
                "recall_weighted": 0.87,
                "f1_weighted": 0.86,
                "confusion_matrix": {},
                "classification_report": {},
                "processing_time": 45.2
            }
        }


class BenchmarkResult(BaseModel):
    """Resultado de benchmark del sistema"""
    total_articles_processed: int = Field(..., description="Total de artículos procesados")
    avg_processing_time: float = Field(..., description="Tiempo promedio por artículo")
    biobert_usage_percent: float = Field(..., description="Porcentaje de uso de BioBERT")
    llm_usage_percent: float = Field(..., description="Porcentaje de uso de LLM")
    accuracy_metrics: dict[str, float] = Field(..., description="Métricas de precisión")
    performance_metrics: dict[str, float] = Field(..., description="Métricas de rendimiento")
    error_rate: float = Field(..., description="Tasa de errores")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "total_articles_processed": 1000,
                "avg_processing_time": 0.85,
                "biobert_usage_percent": 75.5,
                "llm_usage_percent": 24.5,
                "accuracy_metrics": {
                    "overall_accuracy": 0.89,
                    "f1_macro": 0.87
                },
                "performance_metrics": {
                    "throughput_per_hour": 4235,
                    "memory_usage_mb": 1250
                },
                "error_rate": 0.02
            }
        }


class ComparisonRequest(BaseModel):
    """Request para comparar métodos de clasificación"""
    dataset_path: str = Field(..., description="Ruta al dataset para comparación")
    methods: list[str] = Field(..., description="Métodos a comparar", min_items=2)
    confidence_threshold: float = Field(0.7, description="Umbral de confianza", ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "dataset_path": "data/comparison_dataset.csv",
                "methods": ["biobert", "llm", "hybrid"],
                "confidence_threshold": 0.7
            }
        }


class ComparisonResult(BaseModel):
    """Resultado de comparación entre métodos"""
    method_results: dict[str, EvaluationResult] = Field(..., description="Resultados por método")
    comparison_summary: dict[str, Any] = Field(..., description="Resumen comparativo")
    best_method: str = Field(..., description="Mejor método basado en F1-macro")
    recommendations: list[str] = Field(..., description="Recomendaciones basadas en resultados")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "method_results": {
                    "biobert": {"accuracy": 0.85, "f1_macro": 0.82},
                    "hybrid": {"accuracy": 0.89, "f1_macro": 0.87}
                },
                "comparison_summary": {
                    "best_accuracy": "hybrid",
                    "fastest_method": "biobert",
                    "most_consistent": "hybrid"
                },
                "best_method": "hybrid",
                "recommendations": [
                    "Use hybrid method for best accuracy",
                    "Consider biobert for speed-critical applications"
                ]
            }
        }
