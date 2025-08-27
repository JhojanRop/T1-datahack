import json
import os
import time
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class MedicalLLMClassifier:
    """Clasificador LLM usando Gemini REAL"""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.label_names = ['cardiovascular', 'hepatorenal', 'neurological', 'oncological']
        self.request_count = 0
        self.max_requests = 10  # Límite para demo
        self.model = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')  # Modelo más rápido
                self.mode = 'real'
                print("🤖 LLM Gemini REAL inicializado exitosamente")
                print(f"🔑 API Key configurada: ...{self.api_key[-4:]}")
            except Exception as e:
                print(f"❌ Error configurando Gemini: {e}")
                self.mode = 'simulated'
        else:
            self.mode = 'simulated'
            print("⚠️ No se encontró GEMINI_API_KEY, usando modo simulado")

    def classify_complex_case(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificar caso complejo usando Gemini REAL"""

        if self.request_count >= self.max_requests:
            print(f"⚠️ Límite de {self.max_requests} peticiones alcanzado")
            return self._default_prediction()

        self.request_count += 1
        print(f"🤖 Procesando caso {self.request_count}/{self.max_requests} con Gemini...")

        if self.mode == 'real' and self.model:
            return self._classify_with_gemini_real(title, abstract)
        else:
            print("⚠️ Fallback a modo simulado")
            return self._classify_simulated(title, abstract)

    def _classify_with_gemini_real(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificación REAL con Gemini"""

        prompt = f"""
Eres un especialista médico experto en clasificación de literatura científica. Analiza cuidadosamente el siguiente artículo médico y clasifícalo en una o más categorías.

CATEGORÍAS DISPONIBLES:
• cardiovascular: Enfermedades del corazón, vasos sanguíneos, hipertensión, aterosclerosis, infarto
• hepatorenal: Enfermedades del hígado, riñones, insuficiencia renal/hepática, trasplantes
• neurological: Enfermedades del sistema nervioso, cerebro, demencia, epilepsia, neurodegenerativas
• oncological: Cáncer, tumores, oncología, quimioterapia, radioterapia

ARTÍCULO A ANALIZAR:
Título: {title}

Resumen: {abstract}

INSTRUCCIONES:
1. Lee cuidadosamente el título y resumen
2. Identifica las especialidades médicas relevantes
3. Asigna true/false para cada categoría
4. Un artículo puede pertenecer a múltiples categorías
5. Proporciona una confianza entre 0.0 y 1.0

Responde ÚNICAMENTE con un JSON válido en este formato exacto:
{{
    "cardiovascular": true,
    "hepatorenal": false,
    "neurological": false,
    "oncological": true,
    "confidence": 0.92,
    "reasoning": "El artículo trata sobre cardiooncología, combinando aspectos cardiovasculares y oncológicos"
}}
"""

        try:
            # Realizar petición a Gemini con retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Intento {attempt + 1}/{max_retries} - Consultando Gemini...")

                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Baja temperatura para respuestas más consistentes
                            max_output_tokens=500,
                        )
                    )

                    if not response.text:
                        raise ValueError("Respuesta vacía de Gemini")

                    # Procesar respuesta
                    response_text = response.text.strip()
                    print(f"📝 Respuesta Gemini: {response_text[:100]}...")

                    # Extraer JSON
                    json_result = self._extract_json_from_response(response_text)

                    if json_result:
                        print("✅ Respuesta JSON válida obtenida")
                        return {
                            'classification': {
                                'cardiovascular': json_result.get('cardiovascular', False),
                                'hepatorenal': json_result.get('hepatorenal', False),
                                'neurological': json_result.get('neurological', False),
                                'oncological': json_result.get('oncological', False)
                            },
                            'confidence_score': float(json_result.get('confidence', 0.85)),
                            'reasoning': json_result.get('reasoning', 'Análisis Gemini'),
                            'request_number': self.request_count,
                            'model_used': 'gemini-1.5-flash',
                            'attempt': attempt + 1
                        }
                    else:
                        print(f"⚠️ Intento {attempt + 1}: JSON inválido")
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Esperar antes del siguiente intento

                except Exception as e:
                    print(f"❌ Error en intento {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Esperar más tiempo entre reintentos
                    else:
                        raise

            # Si todos los intentos fallan, usar fallback inteligente
            print("⚠️ Todos los intentos fallaron, usando análisis de fallback")
            return self._intelligent_fallback(title, abstract)

        except Exception as e:
            print(f"❌ Error crítico en Gemini: {e}")
            return self._intelligent_fallback(title, abstract)

    def _extract_json_from_response(self, response_text: str) -> dict[str, Any]:
        """Extraer y validar JSON de la respuesta de Gemini"""
        try:
            # Buscar JSON en la respuesta
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No se encontró JSON en la respuesta")

            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)

            # Validar estructura requerida
            required_keys = ['cardiovascular', 'hepatorenal', 'neurological', 'oncological']
            if not all(key in result for key in required_keys):
                raise ValueError("JSON incompleto - faltan categorías")

            # Validar tipos de datos
            for key in required_keys:
                if not isinstance(result[key], bool):
                    result[key] = bool(result[key])  # Convertir a bool si es necesario

            # Validar confianza
            if 'confidence' in result:
                confidence = float(result['confidence'])
                if not 0.0 <= confidence <= 1.0:
                    result['confidence'] = max(0.0, min(1.0, confidence))

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"❌ Error procesando JSON: {e}")
            return None

    def _intelligent_fallback(self, title: str, abstract: str) -> dict[str, Any]:
        """Fallback inteligente cuando Gemini falla"""
        print("🧠 Usando análisis de fallback inteligente...")

        text_combined = f"{title} {abstract}".lower()

        # Diccionarios de palabras clave más completos
        keywords = {
            'cardiovascular': [
                'heart', 'cardiac', 'cardiovascular', 'cardio', 'myocardial', 'coronary',
                'atherosclerosis', 'hypertension', 'arrhythmia', 'valve', 'aortic',
                'ventricular', 'atrial', 'pericardial', 'endocardial', 'vascular'
            ],
            'hepatorenal': [
                'kidney', 'renal', 'liver', 'hepatic', 'nephro', 'hepatorenal',
                'cholecystitis', 'cirrhosis', 'dialysis', 'transplant', 'gallbladder',
                'bile', 'creatinine', 'glomerular', 'hepatitis', 'jaundice'
            ],
            'neurological': [
                'brain', 'neurological', 'neural', 'cerebral', 'neuron', 'dementia',
                'alzheimer', 'parkinson', 'epilepsy', 'seizure', 'stroke', 'migraine',
                'adrenoleukodystrophy', 'encephalitis', 'meningitis', 'cognitive'
            ],
            'oncological': [
                'cancer', 'tumor', 'oncological', 'malignant', 'carcinoma', 'lymphoma',
                'chemotherapy', 'radiation', 'metastasis', 'biopsy', 'oncology',
                'neoplasm', 'leukemia', 'sarcoma', 'adenoma', 'melanoma'
            ]
        }

        classification = {}
        confidence_factors = []

        # Análisis por categoría
        for category, keyword_list in keywords.items():
            matches = sum(1 for keyword in keyword_list if keyword in text_combined)
            has_match = matches > 0
            classification[category] = has_match

            if has_match:
                # Factor de confianza basado en número de coincidencias
                confidence_factor = min(matches / len(keyword_list), 0.3)
                confidence_factors.append(confidence_factor)

        # Si no hay coincidencias, buscar términos médicos generales
        if not any(classification.values()):
            medical_terms = ['patient', 'treatment', 'diagnosis', 'clinical', 'medical', 'therapy']
            if any(term in text_combined for term in medical_terms):
                # Asignar categoría más probable basada en longitud del abstract
                if len(abstract) > 500:
                    classification['oncological'] = True  # Artículos largos suelen ser de oncología
                else:
                    classification['cardiovascular'] = True  # Por defecto
                confidence_factors.append(0.1)

        # Calcular confianza final
        if confidence_factors:
            base_confidence = sum(confidence_factors)
            final_confidence = min(0.95, max(0.6, base_confidence))
        else:
            final_confidence = 0.5

        return {
            'classification': classification,
            'confidence_score': round(final_confidence, 3),
            'reasoning': f"Análisis de fallback: {sum(classification.values())} categorías identificadas",
            'request_number': self.request_count,
            'model_used': 'intelligent_fallback',
            'matches_found': sum(classification.values())
        }

    def _classify_simulated(self, title: str, abstract: str) -> dict[str, Any]:
        """Clasificación simulada básica (último recurso)"""
        import random

        text_combined = f"{title} {abstract}".lower()

        classification = {
            'cardiovascular': 'heart' in text_combined or 'cardiac' in text_combined,
            'hepatorenal': 'kidney' in text_combined or 'liver' in text_combined,
            'neurological': 'brain' in text_combined or 'neural' in text_combined,
            'oncological': 'cancer' in text_combined or 'tumor' in text_combined
        }

        # Si no hay coincidencias, asignar aleatoriamente
        if not any(classification.values()):
            random_category = random.choice(self.label_names)
            classification[random_category] = True

        return {
            'classification': classification,
            'confidence_score': random.uniform(0.6, 0.8),
            'reasoning': "Análisis simulado por palabras clave básicas",
            'request_number': self.request_count,
            'model_used': 'basic_simulation'
        }

    def _default_prediction(self):
        """Predicción por defecto cuando se alcanza el límite"""
        return {
            'classification': {label: False for label in self.label_names},
            'confidence_score': 0.5,
            'reasoning': f"Límite de {self.max_requests} peticiones alcanzado",
            'request_number': self.request_count,
            'model_used': 'default_limit'
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Estadísticas de uso del LLM"""
        return {
            'mode': self.mode,
            'total_requests': self.request_count,
            'max_requests': self.max_requests,
            'remaining_requests': max(0, self.max_requests - self.request_count),
            'api_configured': self.api_key is not None,
            'model_name': 'gemini-1.5-flash' if self.mode == 'real' else 'simulated'
        }

    def reset_usage(self):
        """Reiniciar contador de peticiones"""
        self.request_count = 0
        print("🔄 Contador de peticiones reiniciado")
