"""
Clasificador híbrido que combina BioBERT y LLM de manera inteligente
"""
import asyncio
import time
from statistics import mean
from typing import Any

from core import get_logger, settings

from .biobert_classifier_enhanced import BioBERTClassifierEnhanced
from .llm_classifier_enhanced import LLMClassifierEnhanced

logger = get_logger("hybrid_classifier")


class HybridClassifierEnhanced:
    """Clasificador híbrido inteligente que combina BioBERT y LLM"""

    def __init__(self):
        # Inicializar clasificadores
        self.biobert_classifier = BioBERTClassifierEnhanced()
        self.llm_classifier = LLMClassifierEnhanced()

        # Configuración de routing inteligente
        self.confidence_threshold = settings.confidence_threshold
        self.llm_threshold = settings.llm_threshold

        # Estadísticas de uso
        self.stats = {
            'total_requests': 0,
            'biobert_only': 0,
            'llm_only': 0,
            'hybrid_consensus': 0,
            'biobert_avg_time': [],
            'llm_avg_time': [],
            'hybrid_avg_time': []
        }

        logger.info("Clasificador híbrido inicializado")

    def should_use_llm(self, biobert_result: tuple[list[str], dict[str, float], dict[str, Any]]) -> tuple[bool, str]:
        """Determinar si se debe usar LLM basado en el resultado de BioBERT"""
        domains, confidence_scores, metadata = biobert_result

        max_confidence = max(confidence_scores.values()) if confidence_scores else 0.0
        avg_confidence = mean(confidence_scores.values()) if confidence_scores else 0.0
        entropy = metadata.get('prediction_entropy', 0.0)

        # Criterios para activar LLM
        reasons = []

        # 1. Confianza baja
        if max_confidence < self.llm_threshold:
            reasons.append(f"low_confidence({max_confidence:.3f})")

        # 2. Alta entropía (predicciones dispersas)
        if entropy > 1.0:
            reasons.append(f"high_entropy({entropy:.3f})")

        # 3. Múltiples dominios con confianza similar
        if len(domains) > 1:
            domain_scores = [confidence_scores[d] for d in domains]
            if max(domain_scores) - min(domain_scores) < 0.2:
                reasons.append("similar_confidences")

        # 4. Texto muy corto o muy largo
        text_length = metadata.get('text_length', 0)
        if text_length < 50 or text_length > 2000:
            reasons.append(f"unusual_length({text_length})")

        use_llm = len(reasons) > 0
        reason = ", ".join(reasons) if reasons else "biobert_sufficient"

        return use_llm, reason

    def combine_predictions(self, biobert_result: tuple, llm_result: dict[str, Any]) -> tuple[list[str], dict[str, float], dict[str, Any]]:
        """Combinar predicciones de BioBERT y LLM de manera inteligente"""
        biobert_domains, biobert_scores, biobert_metadata = biobert_result
        llm_domains = llm_result['domains']
        llm_scores = llm_result['confidence_scores']

        # Pesos para combinación (ajustables)
        biobert_weight = 0.6
        llm_weight = 0.4

        # Combinar scores
        combined_scores = {}
        for domain in self.biobert_classifier.medical_domains:
            biobert_score = biobert_scores.get(domain, 0.0)
            llm_score = llm_scores.get(domain, 0.0)

            # Combinación ponderada
            combined_score = (biobert_weight * biobert_score) + (llm_weight * llm_score)
            combined_scores[domain] = combined_score

        # Determinar dominios finales
        final_domains = [
            domain for domain, score in combined_scores.items()
            if score >= self.confidence_threshold
        ]

        # Si no hay dominios, tomar el de mayor confianza
        if not final_domains:
            best_domain = max(combined_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0.1:
                final_domains = [best_domain[0]]

        # Metadatos combinados
        combined_metadata = {
            'biobert_metadata': biobert_metadata,
            'llm_metadata': llm_result.get('metadata', {}),
            'combination_method': 'weighted_average',
            'biobert_weight': biobert_weight,
            'llm_weight': llm_weight,
            'biobert_domains': biobert_domains,
            'llm_domains': llm_domains,
            'llm_reasoning': llm_result.get('reasoning', ''),
            'key_terms': llm_result.get('key_terms', [])
        }

        return final_domains, combined_scores, combined_metadata

    async def classify_article(self, title: str, abstract: str, method: str = "auto") -> dict[str, Any]:
        """Clasificar artículo usando estrategia híbrida inteligente"""
        start_time = time.time()
        self.stats['total_requests'] += 1

        try:
            # Forzar método específico si se solicita
            if method == "biobert":
                return await self._classify_biobert_only(title, abstract)
            elif method == "llm":
                return await self._classify_llm_only(title, abstract)

            # Estrategia híbrida automática
            logger.info("Iniciando clasificación híbrida")

            # Paso 1: BioBERT (rápido)
            biobert_start = time.time()
            biobert_result = self.biobert_classifier.predict_with_confidence_enhanced(f"{title} {abstract}")
            biobert_time = time.time() - biobert_start
            self.stats['biobert_avg_time'].append(biobert_time)

            # Paso 2: Decidir si usar LLM
            use_llm, reason = self.should_use_llm(biobert_result)

            if not use_llm:
                # Solo BioBERT
                self.stats['biobert_only'] += 1
                domains, confidence_scores, metadata = biobert_result

                result = {
                    'domains': domains,
                    'confidence_scores': confidence_scores,
                    'method_used': 'biobert',
                    'processing_time': time.time() - start_time,
                    'metadata': {
                        **metadata,
                        'routing_decision': reason,
                        'llm_triggered': False
                    },
                    'timestamp': time.time()
                }

                logger.info(f"BioBERT-only classification: {domains} (reason: {reason})")
                return result

            # Paso 3: LLM para análisis profundo
            logger.info(f"Activando LLM: {reason}")
            llm_start = time.time()
            llm_result = await self.llm_classifier.classify_complex_case(title, abstract)
            llm_time = time.time() - llm_start
            self.stats['llm_avg_time'].append(llm_time)

            # Paso 4: Combinar resultados
            final_domains, final_scores, combined_metadata = self.combine_predictions(biobert_result, llm_result)
            self.stats['hybrid_consensus'] += 1

            total_time = time.time() - start_time
            self.stats['hybrid_avg_time'].append(total_time)

            result = {
                'domains': final_domains,
                'confidence_scores': final_scores,
                'method_used': 'hybrid',
                'processing_time': total_time,
                'metadata': {
                    **combined_metadata,
                    'routing_decision': reason,
                    'llm_triggered': True,
                    'biobert_time': biobert_time,
                    'llm_time': llm_time
                },
                'timestamp': time.time()
            }

            logger.info(f"Hybrid classification completed: {final_domains} ({total_time:.3f}s)")
            return result

        except Exception as e:
            logger.error(f"Error en clasificación híbrida: {e}")
            # Fallback a BioBERT solo
            try:
                return await self._classify_biobert_only(title, abstract)
            except Exception as fallback_error:
                logger.error(f"Error en fallback: {fallback_error}")
                raise

    async def _classify_biobert_only(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificación solo con BioBERT"""
        start_time = time.time()

        result = self.biobert_classifier.predict_with_confidence_enhanced(f"{title} {abstract}")
        domains, confidence_scores, metadata = result

        return {
            'domains': domains,
            'confidence_scores': confidence_scores,
            'method_used': 'biobert',
            'processing_time': time.time() - start_time,
            'metadata': {
                **metadata,
                'llm_triggered': False
            },
            'timestamp': time.time()
        }

    async def _classify_llm_only(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificación solo con LLM"""
        start_time = time.time()

        llm_result = await self.llm_classifier.classify_complex_case(title, abstract)

        return {
            'domains': llm_result['domains'],
            'confidence_scores': llm_result['confidence_scores'],
            'method_used': 'llm',
            'processing_time': time.time() - start_time,
            'metadata': {
                **llm_result.get('metadata', {}),
                'reasoning': llm_result.get('reasoning', ''),
                'key_terms': llm_result.get('key_terms', []),
                'llm_triggered': True
            },
            'timestamp': time.time()
        }

    async def classify_batch(self, articles: list[dict[str, str]], method: str = "auto",
                           parallel: bool = True) -> list[dict[str, Any]]:
        """Clasificación en lote con procesamiento paralelo opcional"""
        logger.info(f"Iniciando clasificación híbrida en lote: {len(articles)} artículos")

        if parallel and len(articles) > 1:
            # Procesamiento paralelo con límite de concurrencia
            semaphore = asyncio.Semaphore(5)

            async def classify_with_semaphore(article):
                async with semaphore:
                    return await self.classify_article(
                        article.get('title', ''),
                        article.get('abstract', ''),
                        method
                    )

            tasks = [classify_with_semaphore(article) for article in articles]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Procesar excepciones
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error en artículo {i}: {result}")
                    # Resultado fallback
                    fallback = await self._classify_biobert_only(
                        articles[i].get('title', ''),
                        articles[i].get('abstract', '')
                    )
                    fallback['metadata']['error'] = str(result)
                    processed_results.append(fallback)
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # Procesamiento secuencial
            results = []
            for i, article in enumerate(articles):
                try:
                    result = await self.classify_article(
                        article.get('title', ''),
                        article.get('abstract', ''),
                        method
                    )
                    results.append(result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Procesados {i + 1}/{len(articles)} artículos")

                except Exception as e:
                    logger.error(f"Error procesando artículo {i}: {e}")
                    fallback = await self._classify_biobert_only(
                        article.get('title', ''),
                        article.get('abstract', '')
                    )
                    fallback['metadata']['error'] = str(e)
                    results.append(fallback)

            return results

    def get_statistics(self) -> dict[str, Any]:
        """Obtener estadísticas de uso del clasificador híbrido"""
        total = self.stats['total_requests']
        if total == 0:
            return {'message': 'No hay estadísticas disponibles'}

        return {
            'total_requests': total,
            'method_distribution': {
                'biobert_only': self.stats['biobert_only'],
                'llm_only': self.stats['llm_only'],
                'hybrid_consensus': self.stats['hybrid_consensus']
            },
            'method_percentages': {
                'biobert_only': (self.stats['biobert_only'] / total) * 100,
                'llm_only': (self.stats['llm_only'] / total) * 100,
                'hybrid_consensus': (self.stats['hybrid_consensus'] / total) * 100
            },
            'avg_processing_times': {
                'biobert': mean(self.stats['biobert_avg_time']) if self.stats['biobert_avg_time'] else 0,
                'llm': mean(self.stats['llm_avg_time']) if self.stats['llm_avg_time'] else 0,
                'hybrid': mean(self.stats['hybrid_avg_time']) if self.stats['hybrid_avg_time'] else 0
            },
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'llm_threshold': self.llm_threshold
            }
        }

    def adjust_confidence_threshold(self, new_threshold: float):
        """Ajustar umbral de confianza dinámicamente"""
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"Umbral de confianza ajustado: {old_threshold} -> {self.confidence_threshold}")

    def adjust_llm_threshold(self, new_threshold: float):
        """Ajustar umbral para activación de LLM"""
        old_threshold = self.llm_threshold
        self.llm_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"Umbral LLM ajustado: {old_threshold} -> {self.llm_threshold}")

    def reset_statistics(self):
        """Reiniciar estadísticas"""
        self.stats = {
            'total_requests': 0,
            'biobert_only': 0,
            'llm_only': 0,
            'hybrid_consensus': 0,
            'biobert_avg_time': [],
            'llm_avg_time': [],
            'hybrid_avg_time': []
        }
        logger.info("Estadísticas reiniciadas")
