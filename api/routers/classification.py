"""
Router de clasificación médica - Endpoints principales
"""
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from api.auth import User, require_classification_read, require_classification_write
from api.models import (
    ArticleInput,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationResult,
    QuickTestRequest,
)
from core import RequestLogger, log_classification, logger
from services import HybridClassifierEnhanced

router = APIRouter(prefix="/api/v1/classification", tags=["classification"])

# Instancia global del clasificador híbrido
classifier = HybridClassifierEnhanced()


@router.post(
    "/single",
    response_model=ClassificationResult,
    summary="Clasificar un artículo médico",
    description="Clasifica un solo artículo médico en dominios específicos usando el sistema híbrido"
)
async def classify_single_article(
    article: ArticleInput,
    method: str = "auto",
    current_user: User = Depends(require_classification_read)
):
    """Clasificar un artículo médico individual"""
    request_id = str(uuid.uuid4())

    with RequestLogger(request_id, "POST", "/classification/single", current_user.username):
        try:
            # Validar método
            if method not in ["auto", "biobert", "llm", "hybrid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Método debe ser: auto, biobert, llm, o hybrid"
                )

            # Clasificar artículo
            result = await classifier.classify_article(
                title=article.title,
                abstract=article.abstract,
                method=method if method != "auto" else "auto"
            )

            # Log de clasificación
            log_classification(
                request_id=request_id,
                method=result['method_used'],
                domains=result['domains'],
                confidence=max(result['confidence_scores'].values()) if result['confidence_scores'] else 0.0,
                duration=result['processing_time']
            )

            # Formatear respuesta
            classification_result = ClassificationResult(
                domains=result['domains'],
                confidence_scores=result['confidence_scores'],
                method_used=result['method_used'],
                processing_time=result['processing_time'],
                metadata=result['metadata'],
                timestamp=datetime.fromtimestamp(result['timestamp'])
            )

            return classification_result

        except Exception as e:
            logger.error(f"Error en clasificación individual: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en clasificación: {str(e)}"
            )


@router.post(
    "/batch",
    response_model=BatchClassificationResponse,
    summary="Clasificar múltiples artículos",
    description="Clasifica múltiples artículos médicos en lote con procesamiento optimizado"
)
async def classify_batch_articles(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_classification_write)
):
    """Clasificar múltiples artículos en lote"""
    request_id = str(uuid.uuid4())

    with RequestLogger(request_id, "POST", "/classification/batch", current_user.username):
        try:
            start_time = time.time()

            # Preparar datos para clasificación
            articles_data = []
            for article in request.articles:
                articles_data.append({
                    'title': article.title,
                    'abstract': article.abstract
                })

            # Clasificación en lote
            results = await classifier.classify_batch(
                articles=articles_data,
                method=request.method,
                parallel=request.parallel_processing
            )

            # Procesar resultados
            classification_results = []
            successful_count = 0
            errors = []

            for i, result in enumerate(results):
                try:
                    if 'error' in result.get('metadata', {}):
                        errors.append({
                            'index': i,
                            'error': result['metadata']['error'],
                            'article_title': request.articles[i].title[:50] + "..."
                        })
                    else:
                        successful_count += 1

                    classification_result = ClassificationResult(
                        domains=result['domains'],
                        confidence_scores=result['confidence_scores'],
                        method_used=result['method_used'],
                        processing_time=result['processing_time'],
                        metadata=result['metadata'],
                        timestamp=datetime.fromtimestamp(result.get('timestamp', time.time()))
                    )
                    classification_results.append(classification_result)

                except Exception as e:
                    errors.append({
                        'index': i,
                        'error': str(e),
                        'article_title': request.articles[i].title[:50] + "..."
                    })

            total_time = time.time() - start_time

            # Calcular estadísticas de resumen
            if classification_results:
                avg_time = sum(r.processing_time for r in classification_results) / len(classification_results)
                methods_used = {}
                domains_found = {}

                for result in classification_results:
                    # Contar métodos
                    method = result.method_used
                    methods_used[method] = methods_used.get(method, 0) + 1

                    # Contar dominios
                    for domain in result.domains:
                        domains_found[domain] = domains_found.get(domain, 0) + 1
            else:
                avg_time = 0
                methods_used = {}
                domains_found = {}

            summary = {
                'avg_processing_time': avg_time,
                'methods_used': methods_used,
                'domains_found': domains_found,
                'success_rate': (successful_count / len(request.articles)) * 100 if request.articles else 0
            }

            # Log en background
            background_tasks.add_task(
                log_classification,
                request_id=request_id,
                method="batch",
                domains=list(domains_found.keys()),
                confidence=0.0,  # N/A para batch
                duration=total_time
            )

            response = BatchClassificationResponse(
                results=classification_results,
                summary=summary,
                total_processed=len(classification_results),
                total_time=total_time,
                errors=errors
            )

            return response

        except Exception as e:
            logger.error(f"Error en clasificación en lote: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en clasificación en lote: {str(e)}"
            )


@router.post(
    "/quick-test",
    summary="Test rápido de clasificación",
    description="Realiza un test rápido de clasificación para validar el funcionamiento del sistema"
)
async def quick_classification_test(
    request: QuickTestRequest,
    current_user: User = Depends(require_classification_read)
):
    """Test rápido de clasificación"""
    request_id = str(uuid.uuid4())

    try:
        start_time = time.time()

        # Crear artículo temporal para test
        temp_article = ArticleInput(
            title="Quick Test",
            abstract=request.text
        )

        # Clasificar usando método especificado
        result = await classifier.classify_article(
            title=temp_article.title,
            abstract=temp_article.abstract,
            method=request.method
        )

        processing_time = time.time() - start_time

        return {
            "test_result": "success",
            "predicted_domains": result['domains'],
            "confidence_scores": result['confidence_scores'],
            "method_used": result['method_used'],
            "processing_time": processing_time,
            "input_text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error en test rápido: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "test_result": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/methods",
    summary="Métodos de clasificación disponibles",
    description="Lista los métodos de clasificación disponibles y su estado"
)
async def get_classification_methods(
    current_user: User = Depends(require_classification_read)
):
    """Obtener información sobre métodos de clasificación disponibles"""
    try:
        biobert_info = classifier.biobert_classifier.get_model_info()
        llm_info = classifier.llm_classifier.get_service_info()
        hybrid_stats = classifier.get_statistics()

        return {
            "available_methods": [
                {
                    "name": "biobert",
                    "description": "Clasificación rápida usando BioBERT fine-tuned",
                    "status": "active" if biobert_info['model_loaded'] else "inactive",
                    "avg_time": "~0.5s",
                    "domains": biobert_info['domains']
                },
                {
                    "name": "llm",
                    "description": "Análisis profundo usando Gemini LLM",
                    "status": llm_info['service_status'],
                    "avg_time": "~2-5s",
                    "domains": llm_info['domains']
                },
                {
                    "name": "hybrid",
                    "description": "Sistema híbrido inteligente (recomendado)",
                    "status": "active",
                    "avg_time": "~0.5-3s",
                    "domains": biobert_info['domains']
                }
            ],
            "hybrid_statistics": hybrid_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error obteniendo métodos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo información de métodos: {str(e)}"
        )


@router.get(
    "/domains",
    summary="Dominios médicos disponibles",
    description="Lista los dominios médicos que puede identificar el sistema"
)
async def get_medical_domains(
    current_user: User = Depends(require_classification_read)
):
    """Obtener dominios médicos disponibles"""
    try:
        domains_info = {
            "cardiovascular": {
                "description": "Enfermedades del corazón y sistema circulatorio",
                "keywords": ["heart", "cardiac", "cardiovascular", "coronary", "myocardial"],
                "examples": ["myocardial infarction", "coronary artery disease", "heart failure"]
            },
            "neurological": {
                "description": "Enfermedades del sistema nervioso y cerebro",
                "keywords": ["brain", "neural", "neurological", "cognitive", "alzheimer"],
                "examples": ["alzheimer disease", "parkinson", "stroke", "epilepsy"]
            },
            "oncological": {
                "description": "Cáncer y tumores",
                "keywords": ["cancer", "tumor", "oncology", "chemotherapy", "malignant"],
                "examples": ["breast cancer", "lung tumor", "chemotherapy treatment"]
            },
            "hepatorenal": {
                "description": "Enfermedades del hígado y riñones",
                "keywords": ["liver", "kidney", "hepatic", "renal", "dialysis"],
                "examples": ["liver cirrhosis", "kidney failure", "hepatitis"]
            }
        }

        return {
            "total_domains": len(domains_info),
            "domains": domains_info,
            "multilabel_classification": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error obteniendo dominios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo dominios médicos: {str(e)}"
        )
