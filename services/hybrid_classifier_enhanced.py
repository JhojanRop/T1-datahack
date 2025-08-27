from typing import Any

import numpy as np


class HybridMedicalClassifierEnhanced:
    """Sistema híbrido BioBERT + LLM"""

    def __init__(self, biobert_classifier, llm_classifier, confidence_threshold=0.7):
        self.biobert_classifier = biobert_classifier
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold
        self.label_names = biobert_classifier.label_names if biobert_classifier else []

        # Estadísticas
        self.stats = {
            'total_classifications': 0,
            'biobert_used': 0,
            'llm_used': 0,
            'hybrid_decisions': []
        }

        print(f"🔄 Sistema Híbrido inicializado (threshold: {confidence_threshold})")

    def classify_single(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificar un artículo individual"""

        self.stats['total_classifications'] += 1

        # 1. Intentar con BioBERT primero
        biobert_result = self.biobert_classifier.predict_single(title, abstract)
        biobert_confidence = biobert_result['confidence_score']

        # 2. Decidir si usar LLM
        if biobert_confidence >= self.confidence_threshold:
            # Caso obvio - usar BioBERT
            self.stats['biobert_used'] += 1
            decision = 'biobert_confident'
            final_result = biobert_result
        else:
            # Caso difícil - usar LLM
            self.stats['llm_used'] += 1
            decision = 'llm_complex'

            try:
                llm_result = self.llm_classifier.classify_complex_case(title, abstract)
                final_result = llm_result
            except Exception as e:
                print(f"⚠️ Error en LLM, usando BioBERT: {e}")
                final_result = biobert_result
                decision = 'llm_fallback'

        # Registrar decisión
        self.stats['hybrid_decisions'].append({
            'decision': decision,
            'biobert_confidence': biobert_confidence,
            'title': title[:50] + "..." if len(title) > 50 else title
        })

        # Agregar metadatos
        final_result['hybrid_decision'] = decision
        final_result['biobert_confidence'] = biobert_confidence

        return final_result

    def classify_batch(self, articles: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Clasificar lote de artículos"""
        results = []

        for title, abstract in articles:
            result = self.classify_single(title, abstract)
            results.append(result)

        return results

    def get_performance_stats(self) -> dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        total = self.stats['total_classifications']

        if total == 0:
            return {"message": "No hay clasificaciones realizadas"}

        biobert_percentage = (self.stats['biobert_used'] / total) * 100
        llm_percentage = (self.stats['llm_used'] / total) * 100

        # Análisis de confianza
        biobert_confidences = [
            d['biobert_confidence'] for d in self.stats['hybrid_decisions']
        ]

        return {
            'total_classifications': total,
            'model_usage': {
                'biobert_cases': self.stats['biobert_used'],
                'biobert_percentage': round(biobert_percentage, 1),
                'llm_cases': self.stats['llm_used'],
                'llm_percentage': round(llm_percentage, 1)
            },
            'confidence_stats': {
                'mean_biobert_confidence': round(np.mean(biobert_confidences), 3),
                'std_biobert_confidence': round(np.std(biobert_confidences), 3),
                'threshold_used': self.confidence_threshold
            },
            'efficiency': {
                'obvious_cases_ratio': round(biobert_percentage / 100, 2),
                'complex_cases_ratio': round(llm_percentage / 100, 2)
            },
            'recent_decisions': self.stats['hybrid_decisions'][-5:]  # Últimas 5
        }

    def reset_stats(self):
        """Reiniciar estadísticas"""
        self.stats = {
            'total_classifications': 0,
            'biobert_used': 0,
            'llm_used': 0,
            'hybrid_decisions': []
        }
        print("📊 Estadísticas reiniciadas")
