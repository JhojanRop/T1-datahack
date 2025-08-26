"""
Clasificador LLM mejorado con Gemini para casos complejos
"""
import asyncio
import random
import time
from typing import Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from core import get_logger, settings

logger = get_logger("llm_classifier")


class LLMClassifierEnhanced:
    """Clasificador LLM con Gemini para análisis profundo de textos médicos"""

    def __init__(self):
        self.model_name = settings.gemini_model
        self.medical_domains = ['cardiovascular', 'neurological', 'oncological', 'hepatorenal']

        # Configurar Gemini API
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.api_available = True
            logger.info("Gemini API configurada correctamente")
        else:
            self.client = None
            self.api_available = False
            logger.warning("Gemini API key no configurada, usando simulación")

    def create_classification_prompt(self, title: str, abstract: str) -> str:
        """Crear prompt optimizado para clasificación médica"""
        prompt = f"""
Como experto en literatura médica, analiza el siguiente artículo y clasifícalo en uno o más dominios médicos.

ARTÍCULO:
Título: {title}
Abstract: {abstract}

DOMINIOS DISPONIBLES:
1. cardiovascular: Relacionado con corazón, sistema circulatorio, presión arterial, arterias, etc.
2. neurological: Relacionado con cerebro, sistema nervioso, cognición, enfermedades neurodegenerativas, etc.
3. oncological: Relacionado con cáncer, tumores, quimioterapia, radioterapia, oncología, etc.
4. hepatorenal: Relacionado con hígado, riñones, función hepática, función renal, diálisis, etc.

INSTRUCCIONES:
1. Analiza cuidadosamente el contenido médico
2. Identifica los dominios más relevantes (puede ser más de uno)
3. Asigna un score de confianza (0.0-1.0) para cada dominio
4. Considera términos médicos específicos, contexto clínico y temática principal

FORMATO DE RESPUESTA (JSON estricto):
{{
    "domains": ["dominio1", "dominio2"],
    "confidence_scores": {{
        "cardiovascular": 0.0,
        "neurological": 0.0,
        "oncological": 0.0,
        "hepatorenal": 0.0
    }},
    "reasoning": "Breve explicación de la clasificación",
    "key_terms": ["término1", "término2", "término3"]
}}

Responde ÚNICAMENTE con el JSON, sin texto adicional.
"""
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def classify_with_gemini(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificar usando Gemini API con reintentos"""
        try:
            prompt = self.create_classification_prompt(title, abstract)

            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                    candidate_count=1
                )
            )

            # Parsear respuesta JSON
            import json
            result = json.loads(response.text.strip())

            # Validar estructura
            required_keys = ['domains', 'confidence_scores', 'reasoning', 'key_terms']
            if not all(key in result for key in required_keys):
                raise ValueError("Respuesta de Gemini con formato incorrecto")

            # Validar dominios
            valid_domains = [d for d in result['domains'] if d in self.medical_domains]
            result['domains'] = valid_domains

            # Validar scores de confianza
            for domain in self.medical_domains:
                if domain not in result['confidence_scores']:
                    result['confidence_scores'][domain] = 0.0
                else:
                    score = result['confidence_scores'][domain]
                    result['confidence_scores'][domain] = max(0.0, min(1.0, float(score)))

            logger.info(f"Gemini classification successful: {valid_domains}")
            return result

        except Exception as e:
            logger.error(f"Error en clasificación Gemini: {e}")
            raise

    def simulate_llm_response_enhanced(self, title: str, abstract: str) -> dict[str, Any]:
        """Simulación mejorada de respuesta LLM cuando API no está disponible"""
        text = f"{title} {abstract}".lower()

        # Análisis heurístico basado en palabras clave
        domain_scores = {
            'cardiovascular': 0.0,
            'neurological': 0.0,
            'oncological': 0.0,
            'hepatorenal': 0.0
        }

        # Palabras clave específicas para simulación
        keyword_weights = {
            'cardiovascular': {
                'heart': 0.9, 'cardiac': 0.9, 'cardiovascular': 0.95, 'coronary': 0.8,
                'myocardial': 0.85, 'artery': 0.7, 'blood pressure': 0.8, 'hypertension': 0.75,
                'ecg': 0.7, 'stent': 0.8, 'valve': 0.75, 'angina': 0.8
            },
            'neurological': {
                'brain': 0.9, 'neural': 0.85, 'neurological': 0.95, 'cognitive': 0.8,
                'alzheimer': 0.9, 'parkinson': 0.9, 'stroke': 0.85, 'seizure': 0.8,
                'epilepsy': 0.85, 'dementia': 0.85, 'mri': 0.7, 'eeg': 0.75
            },
            'oncological': {
                'cancer': 0.95, 'tumor': 0.9, 'oncology': 0.95, 'chemotherapy': 0.9,
                'radiation': 0.8, 'metastasis': 0.9, 'malignant': 0.85, 'carcinoma': 0.9,
                'lymphoma': 0.9, 'leukemia': 0.9, 'biopsy': 0.8, 'therapeutic': 0.7
            },
            'hepatorenal': {
                'liver': 0.9, 'kidney': 0.9, 'hepatic': 0.9, 'renal': 0.9,
                'dialysis': 0.85, 'cirrhosis': 0.85, 'hepatitis': 0.85, 'transplant': 0.8,
                'creatinine': 0.8, 'nephrology': 0.9, 'glomerular': 0.8, 'filtration': 0.75
            }
        }

        found_terms = {}

        # Calcular scores por dominio
        for domain, keywords in keyword_weights.items():
            domain_terms = []
            total_weight = 0

            for keyword, weight in keywords.items():
                if keyword in text:
                    domain_scores[domain] += weight
                    total_weight += weight
                    domain_terms.append(keyword)

            # Normalizar score
            if total_weight > 0:
                domain_scores[domain] = min(domain_scores[domain] / len(keywords), 1.0)

            found_terms[domain] = domain_terms

        # Agregar algo de variabilidad realista
        for domain in domain_scores:
            noise = random.uniform(-0.05, 0.05)
            domain_scores[domain] = max(0.0, min(1.0, domain_scores[domain] + noise))

        # Determinar dominios predichos
        threshold = 0.3
        predicted_domains = [
            domain for domain, score in domain_scores.items()
            if score >= threshold
        ]

        # Si no hay predicciones, tomar la de mayor score
        if not predicted_domains:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0.1:
                predicted_domains = [best_domain[0]]

        # Encontrar términos clave generales
        all_terms = []
        for domain_terms in found_terms.values():
            all_terms.extend(domain_terms)
        key_terms = list(set(all_terms))[:5]  # Top 5 términos únicos

        # Generar explicación
        if predicted_domains:
            reasoning = f"Clasificado como {', '.join(predicted_domains)} basado en términos médicos específicos encontrados."
        else:
            reasoning = "No se encontraron términos médicos específicos suficientes para una clasificación confiable."

        return {
            'domains': predicted_domains,
            'confidence_scores': domain_scores,
            'reasoning': reasoning,
            'key_terms': key_terms,
            'simulation_mode': True
        }

    async def classify_complex_case(self, title: str, abstract: str, **kwargs) -> dict[str, Any]:
        """Clasificar caso complejo usando LLM"""
        start_time = time.time()

        try:
            if self.api_available:
                result = await self.classify_with_gemini(title, abstract)
            else:
                logger.info("Usando simulación LLM (API no disponible)")
                result = self.simulate_llm_response_enhanced(title, abstract)

            processing_time = time.time() - start_time

            # Agregar metadatos
            result['metadata'] = {
                'processing_time': processing_time,
                'method': 'gemini' if self.api_available else 'simulation',
                'model_name': self.model_name,
                'api_available': self.api_available,
                'text_length': len(f"{title} {abstract}")
            }

            logger.info(f"LLM classification completed in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error en clasificación LLM: {e}")
            # Fallback a simulación en caso de error
            fallback_result = self.simulate_llm_response_enhanced(title, abstract)
            fallback_result['metadata'] = {
                'processing_time': time.time() - start_time,
                'method': 'fallback_simulation',
                'error': str(e),
                'api_available': False
            }
            return fallback_result

    async def batch_classify(self, articles: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Clasificación en lote asíncrona"""
        logger.info(f"Iniciando clasificación LLM en lote: {len(articles)} artículos")

        # Limitar concurrencia para evitar rate limits
        semaphore = asyncio.Semaphore(3)

        async def classify_with_semaphore(article):
            async with semaphore:
                return await self.classify_complex_case(
                    article.get('title', ''),
                    article.get('abstract', '')
                )

        tasks = [classify_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Procesar resultados y manejar excepciones
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error en artículo {i}: {result}")
                # Resultado fallback
                fallback = self.simulate_llm_response_enhanced(
                    articles[i].get('title', ''),
                    articles[i].get('abstract', '')
                )
                fallback['metadata']['error'] = str(result)
                processed_results.append(fallback)
            else:
                processed_results.append(result)

        logger.info(f"Clasificación LLM en lote completada: {len(processed_results)} resultados")
        return processed_results

    def get_service_info(self) -> dict[str, Any]:
        """Información del servicio LLM"""
        return {
            'model_name': self.model_name,
            'api_available': self.api_available,
            'domains': self.medical_domains,
            'service_status': 'active' if self.api_available else 'simulation_mode'
        }
