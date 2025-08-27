import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


class MedicalClassificationPipelineEnhanced:
    """Pipeline completo para clasificación médica en producción"""

    def __init__(self, hybrid_classifier, preprocessor, confidence_threshold: float = 0.7):
        self.hybrid_classifier = hybrid_classifier
        self.preprocessor = preprocessor
        self.confidence_threshold = confidence_threshold

        # Métricas del pipeline
        self.pipeline_stats = {
            'total_processed': 0,
            'processing_times': [],
            'error_count': 0,
            'start_time': datetime.now()
        }

        # Logger
        self.logger = self._setup_logger()

        print(f"🚀 Pipeline médico inicializado (threshold: {confidence_threshold})")

    def _setup_logger(self) -> logging.Logger:
        """Configurar logger para el pipeline"""
        logger = logging.getLogger('MedicalPipeline')
        logger.setLevel(logging.INFO)

        # Handler para consola
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def process_single_article(self, title: str, abstract: str,
                              include_metadata: bool = True) -> dict[str, Any]:
        """Procesar un artículo individual"""

        start_time = time.time()
        self.pipeline_stats['total_processed'] += 1

        try:
            # 1. Preprocesamiento
            title_clean = self.preprocessor.clean_medical_text(title)
            abstract_clean = self.preprocessor.clean_medical_text(abstract)

            # 2. Clasificación híbrida
            result = self.hybrid_classifier.classify_single(title_clean, abstract_clean)

            # 3. Post-procesamiento
            processed_result = self._post_process_result(result, title, abstract)

            # 4. Agregar metadatos si se solicita
            if include_metadata:
                processed_result['metadata'] = self._generate_metadata(
                    title, abstract, title_clean, abstract_clean, start_time
                )

            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self.pipeline_stats['processing_times'].append(processing_time)

            self.logger.info(f"Artículo procesado exitosamente en {processing_time:.3f}s")

            return processed_result

        except Exception as e:
            self.pipeline_stats['error_count'] += 1
            self.logger.error(f"Error procesando artículo: {e}")

            return {
                'classification': {label: False for label in self.hybrid_classifier.label_names},
                'confidence_score': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def process_batch(self, articles: list[tuple[str, str]],
                     batch_size: int = 10,
                     include_progress: bool = True) -> list[dict[str, Any]]:
        """Procesar lote de artículos"""

        self.logger.info(f"Iniciando procesamiento de {len(articles)} artículos en lotes de {batch_size}")

        results = []
        total_batches = (len(articles) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(articles), batch_size):
            batch_articles = articles[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1

            if include_progress:
                print(f"🔄 Procesando lote {batch_num}/{total_batches} ({len(batch_articles)} artículos)")

            # Procesar lote actual
            batch_results = []
            for title, abstract in batch_articles:
                result = self.process_single_article(title, abstract, include_metadata=False)
                batch_results.append(result)

            results.extend(batch_results)

            # Pequeña pausa entre lotes para evitar sobrecarga
            if batch_num < total_batches:
                time.sleep(0.1)

        self.logger.info(f"Procesamiento completado: {len(results)} artículos procesados")
        return results

    def process_dataframe(self, df: pd.DataFrame,
                         title_col: str = 'title',
                         abstract_col: str = 'abstract') -> pd.DataFrame:
        """Procesar DataFrame completo"""

        self.logger.info(f"Procesando DataFrame con {len(df)} filas")

        # Preparar artículos
        articles = [(row[title_col], row[abstract_col]) for _, row in df.iterrows()]

        # Procesar en lotes
        results = self.process_batch(articles)

        # Convertir resultados a DataFrame
        results_df = self._results_to_dataframe(results, df.index)

        # Combinar con DataFrame original
        combined_df = pd.concat([df, results_df], axis=1)

        return combined_df

    def _post_process_result(self, result: dict[str, Any],
                           original_title: str, original_abstract: str) -> dict[str, Any]:
        """Post-procesar resultado de clasificación"""

        # Validar clasificación
        classification = result.get('classification', {})

        # Asegurar que todas las etiquetas estén presentes
        for label in self.hybrid_classifier.label_names:
            if label not in classification:
                classification[label] = False

        # Calcular scores adicionales
        confidence_score = result.get('confidence_score', 0.0)

        # Categorizar confianza
        confidence_category = self._categorize_confidence(confidence_score)

        # Contar etiquetas predichas
        predicted_labels = [label for label, value in classification.items() if value]

        return {
            'classification': classification,
            'confidence_score': float(confidence_score),
            'confidence_category': confidence_category,
            'predicted_labels': predicted_labels,
            'num_predicted_labels': len(predicted_labels),
            'reasoning': result.get('reasoning', ''),
            'model_used': result.get('model_used', 'unknown'),
            'hybrid_decision': result.get('hybrid_decision', 'unknown')
        }

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorizar nivel de confianza"""
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        elif confidence >= 0.3:
            return 'low'
        else:
            return 'very_low'

    def _generate_metadata(self, original_title: str, original_abstract: str,
                          clean_title: str, clean_abstract: str,
                          start_time: float) -> dict[str, Any]:
        """Generar metadatos del procesamiento"""

        processing_time = time.time() - start_time

        return {
            'processing_time': round(processing_time, 4),
            'timestamp': datetime.now().isoformat(),
            'original_lengths': {
                'title': len(original_title),
                'abstract': len(original_abstract)
            },
            'clean_lengths': {
                'title': len(clean_title),
                'abstract': len(clean_abstract)
            },
            'text_reduction': {
                'title': len(original_title) - len(clean_title),
                'abstract': len(original_abstract) - len(clean_abstract)
            },
            'pipeline_version': '2.0'
        }

    def _results_to_dataframe(self, results: list[dict[str, Any]],
                             index: pd.Index) -> pd.DataFrame:
        """Convertir resultados a DataFrame"""

        # Extraer clasificaciones
        classifications = [result['classification'] for result in results]

        # Crear DataFrame de clasificaciones
        class_df = pd.DataFrame(classifications, index=index)

        # Agregar columnas adicionales
        additional_cols = {
            'confidence_score': [result['confidence_score'] for result in results],
            'confidence_category': [result['confidence_category'] for result in results],
            'num_predicted_labels': [result['num_predicted_labels'] for result in results],
            'model_used': [result['model_used'] for result in results],
            'hybrid_decision': [result['hybrid_decision'] for result in results]
        }

        additional_df = pd.DataFrame(additional_cols, index=index)

        # Combinar
        results_df = pd.concat([class_df, additional_df], axis=1)

        return results_df

    def evaluate_pipeline_performance(self, df_with_predictions: pd.DataFrame,
                                    true_labels_cols: list[str]) -> dict[str, Any]:
        """Evaluar rendimiento del pipeline"""

        from utils.medical_evaluator import MedicalEvaluatorEnhanced

        # Preparar datos verdaderos y predichos
        y_true = df_with_predictions[true_labels_cols].values
        y_pred = df_with_predictions[self.hybrid_classifier.label_names].values

        # Crear evaluador
        evaluator = MedicalEvaluatorEnhanced(self.hybrid_classifier.label_names)

        # Realizar evaluación
        evaluation_results = evaluator.evaluate_predictions(y_true, y_pred)

        # Agregar estadísticas del pipeline
        evaluation_results['pipeline_stats'] = self.get_pipeline_statistics()

        return evaluation_results

    def get_pipeline_statistics(self) -> dict[str, Any]:
        """Obtener estadísticas del pipeline"""

        processing_times = self.pipeline_stats['processing_times']
        uptime = (datetime.now() - self.pipeline_stats['start_time']).total_seconds()

        stats = {
            'total_processed': self.pipeline_stats['total_processed'],
            'error_count': self.pipeline_stats['error_count'],
            'error_rate': self.pipeline_stats['error_count'] / max(self.pipeline_stats['total_processed'], 1),
            'uptime_seconds': uptime,
            'throughput_per_second': self.pipeline_stats['total_processed'] / max(uptime, 1)
        }

        if processing_times:
            stats.update({
                'avg_processing_time': np.mean(processing_times),
                'median_processing_time': np.median(processing_times),
                'min_processing_time': min(processing_times),
                'max_processing_time': max(processing_times),
                'std_processing_time': np.std(processing_times)
            })

        # Estadísticas del clasificador híbrido
        if hasattr(self.hybrid_classifier, 'get_performance_stats'):
            stats['hybrid_classifier_stats'] = self.hybrid_classifier.get_performance_stats()

        return stats

    def reset_statistics(self):
        """Reiniciar estadísticas del pipeline"""
        self.pipeline_stats = {
            'total_processed': 0,
            'processing_times': [],
            'error_count': 0,
            'start_time': datetime.now()
        }

        # Reiniciar estadísticas del clasificador híbrido
        if hasattr(self.hybrid_classifier, 'reset_stats'):
            self.hybrid_classifier.reset_stats()

        self.logger.info("Estadísticas del pipeline reiniciadas")

    def export_results(self, df_with_predictions: pd.DataFrame,
                      output_path: str, format: str = 'csv'):
        """Exportar resultados del pipeline"""

        try:
            if format.lower() == 'csv':
                df_with_predictions.to_csv(output_path, index=False)
            elif format.lower() == 'excel':
                df_with_predictions.to_excel(output_path, index=False)
            elif format.lower() == 'json':
                df_with_predictions.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Formato no soportado: {format}")

            self.logger.info(f"Resultados exportados a: {output_path}")

        except Exception as e:
            self.logger.error(f"Error exportando resultados: {e}")
            raise

    def get_pipeline_health(self) -> dict[str, Any]:
        """Obtener estado de salud del pipeline"""

        stats = self.get_pipeline_statistics()

        # Determinar salud basada en métricas
        health_score = 1.0
        health_issues = []

        # Verificar tasa de errores
        if stats['error_rate'] > 0.1:  # Más del 10% de errores
            health_score -= 0.3
            health_issues.append("Alta tasa de errores")

        # Verificar tiempos de procesamiento
        if 'avg_processing_time' in stats and stats['avg_processing_time'] > 5.0:
            health_score -= 0.2
            health_issues.append("Tiempos de procesamiento lentos")

        # Verificar throughput
        if stats['throughput_per_second'] < 0.1:
            health_score -= 0.2
            health_issues.append("Bajo throughput")

        # Categorizar salud
        if health_score >= 0.8:
            health_status = "excellent"
        elif health_score >= 0.6:
            health_status = "good"
        elif health_score >= 0.4:
            health_status = "fair"
        else:
            health_status = "poor"

        return {
            'health_status': health_status,
            'health_score': round(health_score, 2),
            'health_issues': health_issues,
            'statistics': stats,
            'recommendations': self._generate_health_recommendations(health_issues)
        }

    def _generate_health_recommendations(self, issues: list[str]) -> list[str]:
        """Generar recomendaciones basadas en problemas de salud"""

        recommendations = []

        if "Alta tasa de errores" in issues:
            recommendations.append("Revisar calidad de datos de entrada")
            recommendations.append("Verificar configuración de modelos")

        if "Tiempos de procesamiento lentos" in issues:
            recommendations.append("Considerar optimización de modelos")
            recommendations.append("Reducir tamaño de lotes")

        if "Bajo throughput" in issues:
            recommendations.append("Aumentar tamaño de lotes")
            recommendations.append("Optimizar pipeline de preprocesamiento")

        if not recommendations:
            recommendations.append("Pipeline funcionando correctamente")

        return recommendations
