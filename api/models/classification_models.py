"""
Modelos Pydantic para endpoints de clasificación médica
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class ArticleInput(BaseModel):
    """Input para clasificación de un artículo médico"""
    title: str = Field(..., description="Título del artículo médico", min_length=5, max_length=500)
    abstract: str = Field(..., description="Abstract del artículo médico", min_length=20, max_length=5000)
    authors: str | None = Field(None, description="Autores del artículo")
    journal: str | None = Field(None, description="Revista de publicación")
    keywords: list[str] | None = Field(None, description="Palabras clave del artículo")

    @validator('title', 'abstract')
    def validate_text_content(cls, v):
        if not v or not v.strip():
            raise ValueError("El contenido no puede estar vacío")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "title": "Effects of ACE inhibitors on cardiovascular outcomes",
                "abstract": "This study examines the cardiovascular benefits of ACE inhibitors in patients with hypertension...",
                "authors": "Smith J., Johnson A., Brown K.",
                "journal": "Journal of Cardiology",
                "keywords": ["ACE inhibitors", "cardiovascular", "hypertension"]
            }
        }


class ClassificationResult(BaseModel):
    """Resultado de clasificación de un artículo"""
    domains: list[str] = Field(..., description="Dominios médicos identificados")
    confidence_scores: dict[str, float] = Field(..., description="Scores de confianza por dominio")
    method_used: str = Field(..., description="Método utilizado (biobert/llm/hybrid)")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp del procesamiento")

    class Config:
        schema_extra = {
            "example": {
                "domains": ["cardiovascular", "neurological"],
                "confidence_scores": {
                    "cardiovascular": 0.89,
                    "neurological": 0.76,
                    "oncological": 0.12,
                    "hepatorenal": 0.08
                },
                "method_used": "hybrid",
                "processing_time": 1.23,
                "metadata": {
                    "biobert_confidence": 0.85,
                    "llm_triggered": True,
                    "threshold_used": 0.7
                },
                "timestamp": "2025-08-26T10:30:00"
            }
        }


class BatchClassificationRequest(BaseModel):
    """Request para clasificación en lote"""
    articles: list[ArticleInput] = Field(..., description="Lista de artículos a clasificar", min_items=1, max_items=100)
    method: str | None = Field("hybrid", description="Método a utilizar: biobert, llm, hybrid")
    confidence_threshold: float | None = Field(0.7, description="Umbral de confianza", ge=0.0, le=1.0)
    parallel_processing: bool | None = Field(True, description="Procesamiento en paralelo")

    @validator('method')
    def validate_method(cls, v):
        if v not in ['biobert', 'llm', 'hybrid']:
            raise ValueError("Método debe ser 'biobert', 'llm' o 'hybrid'")
        return v

    class Config:
        schema_extra = {
            "example": {
                "articles": [
                    {
                        "title": "Cardiovascular effects of new drug",
                        "abstract": "Study on cardiovascular outcomes..."
                    }
                ],
                "method": "hybrid",
                "confidence_threshold": 0.7,
                "parallel_processing": True
            }
        }


class BatchClassificationResponse(BaseModel):
    """Response para clasificación en lote"""
    results: list[ClassificationResult] = Field(..., description="Resultados de clasificación")
    summary: dict[str, Any] = Field(..., description="Resumen del batch")
    total_processed: int = Field(..., description="Total de artículos procesados")
    total_time: float = Field(..., description="Tiempo total de procesamiento")
    errors: list[dict[str, Any]] = Field(default_factory=list, description="Errores encontrados")

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "domains": ["cardiovascular"],
                        "confidence_scores": {"cardiovascular": 0.89},
                        "method_used": "biobert",
                        "processing_time": 0.45
                    }
                ],
                "summary": {
                    "avg_processing_time": 0.45,
                    "methods_used": {"biobert": 8, "llm": 2},
                    "domains_found": {"cardiovascular": 5, "neurological": 3}
                },
                "total_processed": 10,
                "total_time": 4.5,
                "errors": []
            }
        }


class QuickTestRequest(BaseModel):
    """Request para test rápido de clasificación"""
    text: str = Field(..., description="Texto médico para clasificar", min_length=10, max_length=1000)
    method: str | None = Field("biobert", description="Método a utilizar para test rápido")

    @validator('method')
    def validate_method(cls, v):
        if v not in ['biobert', 'llm', 'hybrid']:
            raise ValueError("Método debe ser 'biobert', 'llm' o 'hybrid'")
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "Patient with acute myocardial infarction treated with stent placement",
                "method": "biobert"
            }
        }
