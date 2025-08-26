"""
Clasificador BioBERT mejorado para textos médicos
"""
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core import get_logger, settings

logger = get_logger("biobert_classifier")


class BioBERTClassifierEnhanced:
    """Clasificador BioBERT con capacidades mejoradas de confianza y análisis"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.biobert_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        # Dominios médicos objetivo
        self.medical_domains = {
            'cardiovascular': 0,
            'neurological': 1,
            'oncological': 2,
            'hepatorenal': 3
        }

        # Palabras clave por dominio para análisis de confianza
        self.domain_keywords = {
            'cardiovascular': [
                'heart', 'cardiac', 'cardiovascular', 'coronary', 'myocardial', 'artery', 'blood pressure',
                'hypertension', 'atherosclerosis', 'angina', 'valve', 'echocardiography', 'ECG', 'stent'
            ],
            'neurological': [
                'brain', 'neural', 'neurological', 'cognitive', 'memory', 'alzheimer', 'parkinson',
                'stroke', 'seizure', 'epilepsy', 'dementia', 'neurodegenerative', 'MRI', 'EEG'
            ],
            'oncological': [
                'cancer', 'tumor', 'oncology', 'chemotherapy', 'radiation', 'metastasis', 'malignant',
                'benign', 'biopsy', 'carcinoma', 'lymphoma', 'leukemia', 'oncogene', 'therapeutic'
            ],
            'hepatorenal': [
                'liver', 'kidney', 'hepatic', 'renal', 'nephrology', 'dialysis', 'cirrhosis',
                'hepatitis', 'transplant', 'creatinine', 'urea', 'filtration', 'nephron', 'glomerular'
            ]
        }

        self.load_model()

    def load_model(self):
        """Cargar modelo BioBERT"""
        try:
            logger.info(f"Cargando modelo BioBERT desde {self.model_path}")

            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Modelo no encontrado en {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Modelo BioBERT cargado exitosamente en {self.device}")

        except Exception as e:
            logger.error(f"Error cargando modelo BioBERT: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocesar texto médico"""
        # Limpiar y normalizar texto
        text = text.strip().lower()

        # Remover caracteres especiales pero mantener puntuación médica
        import re
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def calculate_keyword_confidence(self, text: str) -> dict[str, float]:
        """Calcular confianza basada en palabras clave del dominio"""
        text_lower = text.lower()
        confidences = {}

        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = min(matches / len(keywords), 1.0)
            confidences[domain] = confidence

        return confidences

    def calculate_confidence_scores_robust(self, logits: torch.Tensor, text: str) -> dict[str, float]:
        """Calcular scores de confianza robustos usando múltiples métodos"""
        # Método 1: Softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        softmax_scores = probs.cpu().numpy().flatten()

        # Método 2: Entropía normalizada (menor entropía = mayor confianza)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = np.log(len(self.medical_domains))
        normalized_entropy = 1.0 - (entropy.item() / max_entropy)

        # Método 3: Diferencia entre top-2 predicciones
        sorted_probs, _ = torch.sort(probs, descending=True)
        top2_diff = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()

        # Método 4: Confianza basada en keywords
        keyword_confidences = self.calculate_keyword_confidence(text)

        # Combinar métodos para score final por dominio
        final_scores = {}
        domain_names = list(self.medical_domains.keys())

        for i, domain in enumerate(domain_names):
            softmax_conf = softmax_scores[i]
            keyword_conf = keyword_confidences[domain]

            # Peso combinado (70% modelo, 30% keywords)
            combined_conf = 0.7 * softmax_conf + 0.3 * keyword_conf

            # Aplicar modificadores basados en entropía y top2_diff
            entropy_modifier = normalized_entropy
            top2_modifier = top2_diff if i == np.argmax(softmax_scores) else 1.0

            final_conf = combined_conf * entropy_modifier * top2_modifier
            final_scores[domain] = float(np.clip(final_conf, 0.0, 1.0))

        return final_scores

    def predict_with_confidence_enhanced(self, text: str, threshold: float = None) -> tuple[list[str], dict[str, float], dict[str, Any]]:
        """Predecir dominios con análisis de confianza mejorado"""
        start_time = time.time()
        threshold = threshold or settings.confidence_threshold

        try:
            # Preprocesar texto
            processed_text = self.preprocess_text(text)

            # Tokenizar
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Calcular scores de confianza robustos
            confidence_scores = self.calculate_confidence_scores_robust(logits, text)

            # Filtrar dominios por umbral
            predicted_domains = [
                domain for domain, score in confidence_scores.items()
                if score >= threshold
            ]

            # Si no hay predicciones, tomar la de mayor confianza
            if not predicted_domains:
                best_domain = max(confidence_scores.items(), key=lambda x: x[1])
                if best_domain[1] > 0.1:  # Umbral mínimo muy bajo
                    predicted_domains = [best_domain[0]]

            processing_time = time.time() - start_time

            # Metadatos adicionales
            metadata = {
                'processing_time': processing_time,
                'text_length': len(text),
                'processed_text_length': len(processed_text),
                'max_confidence': max(confidence_scores.values()),
                'min_confidence': min(confidence_scores.values()),
                'avg_confidence': np.mean(list(confidence_scores.values())),
                'threshold_used': threshold,
                'device_used': str(self.device),
                'prediction_entropy': self._calculate_entropy(confidence_scores)
            }

            logger.info(f"BioBERT prediction completed: domains={predicted_domains}, max_conf={metadata['max_confidence']:.3f}")

            return predicted_domains, confidence_scores, metadata

        except Exception as e:
            logger.error(f"Error en predicción BioBERT: {e}")
            raise

    def _calculate_entropy(self, scores: dict[str, float]) -> float:
        """Calcular entropía de las predicciones"""
        values = list(scores.values())
        # Normalizar para que sumen 1
        total = sum(values)
        if total == 0:
            return 0.0

        probs = [v/total for v in values]
        entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
        return entropy

    def batch_predict(self, texts: list[str], threshold: float = None) -> list[tuple[list[str], dict[str, float], dict[str, Any]]]:
        """Predicción en lote para múltiples textos"""
        logger.info(f"Iniciando predicción en lote para {len(texts)} textos")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict_with_confidence_enhanced(text, threshold)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Procesados {i + 1}/{len(texts)} textos")

            except Exception as e:
                logger.error(f"Error procesando texto {i}: {e}")
                # Resultado por defecto en caso de error
                results.append(([], {domain: 0.0 for domain in self.medical_domains}, {'error': str(e)}))

        logger.info(f"Predicción en lote completada: {len(results)} resultados")
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Información del modelo cargado"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'domains': list(self.medical_domains.keys()),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0
        }
