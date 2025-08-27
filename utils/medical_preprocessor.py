import re
from typing import Any

import pandas as pd


class MedicalTextPreprocessor:
    """Preprocesador de texto médico basado en el notebook"""

    def __init__(self):
        # Palabras de parada médicas específicas
        self.medical_stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among'
        }

        # Patrones de limpieza
        self.cleaning_patterns = [
            (r'\b\d+\s*mg\b', 'DOSAGE'),  # Dosis de medicamentos
            (r'\b\d+\s*ml\b', 'VOLUME'),  # Volúmenes
            (r'\b\d+\s*%\b', 'PERCENTAGE'),  # Porcentajes
            (r'\bp\s*<\s*0\.0\d+\b', 'PVALUE'),  # P-values
            (r'\bn\s*=\s*\d+\b', 'SAMPLESIZE'),  # Tamaños de muestra
        ]

        print("🧹 Preprocesador médico inicializado")

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesar dataset completo"""
        print(f"🔄 Preprocesando {len(df)} artículos...")

        df_processed = df.copy()

        # Limpiar títulos y abstracts
        df_processed['title_clean'] = df_processed['title'].apply(self.clean_medical_text)
        df_processed['abstract_clean'] = df_processed['abstract'].apply(self.clean_medical_text)

        # Combinar texto limpio
        df_processed['combined_text'] = (
            df_processed['title_clean'] + ' [SEP] ' + df_processed['abstract_clean']
        )

        # Limpiar etiquetas
        df_processed['group_clean'] = df_processed['group'].apply(self.clean_labels)

        # Estadísticas de limpieza
        original_chars = df['title'].str.len().sum() + df['abstract'].str.len().sum()
        clean_chars = df_processed['title_clean'].str.len().sum() + df_processed['abstract_clean'].str.len().sum()
        reduction = ((original_chars - clean_chars) / original_chars) * 100

        print("✅ Preprocesamiento completado:")
        print(f"   📊 Reducción de caracteres: {reduction:.1f}%")
        print(f"   📝 Longitud promedio título: {df_processed['title_clean'].str.len().mean():.1f}")
        print(f"   📄 Longitud promedio abstract: {df_processed['abstract_clean'].str.len().mean():.1f}")

        return df_processed

    def clean_medical_text(self, text: str) -> str:
        """Limpiar texto médico individual"""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convertir a minúsculas
        text = text.lower()

        # Aplicar patrones de limpieza específicos
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Limpiar caracteres especiales pero preservar algunos importantes
        text = re.sub(r'[^\w\s\-\[\]]', ' ', text)

        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)

        # Remover espacios al inicio y final
        text = text.strip()

        return text

    def clean_labels(self, labels: str) -> str:
        """Limpiar etiquetas médicas"""
        if pd.isna(labels) or not isinstance(labels, str):
            return ""

        # Separar por | y limpiar cada etiqueta
        label_list = [label.strip().lower() for label in str(labels).split('|')]

        # Filtrar etiquetas vacías
        label_list = [label for label in label_list if label]

        # Normalizar nombres de etiquetas comunes
        label_mapping = {
            'cardio': 'cardiovascular',
            'cardiac': 'cardiovascular',
            'heart': 'cardiovascular',
            'neuro': 'neurological',
            'neural': 'neurological',
            'brain': 'neurological',
            'onco': 'oncological',
            'cancer': 'oncological',
            'tumor': 'oncological',
            'hepato': 'hepatorenal',
            'renal': 'hepatorenal',
            'kidney': 'hepatorenal',
            'liver': 'hepatorenal'
        }

        normalized_labels = []
        for label in label_list:
            normalized = label_mapping.get(label, label)
            if normalized not in normalized_labels:
                normalized_labels.append(normalized)

        return '|'.join(normalized_labels)

    def extract_medical_features(self, text: str) -> dict[str, Any]:
        """Extraer características médicas específicas"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_dosage': bool(re.search(r'\b\d+\s*(mg|ml|g)\b', text, re.IGNORECASE)),
            'has_pvalue': bool(re.search(r'\bp\s*[<>=]\s*0\.\d+', text, re.IGNORECASE)),
            'has_sample_size': bool(re.search(r'\bn\s*=\s*\d+', text, re.IGNORECASE)),
            'has_percentage': bool(re.search(r'\d+\s*%', text)),
            'medical_term_density': self._calculate_medical_density(text)
        }

        return features

    def _calculate_medical_density(self, text: str) -> float:
        """Calcular densidad de términos médicos"""
        medical_terms = [
            'patient', 'treatment', 'therapy', 'diagnosis', 'clinical', 'medical',
            'syndrome', 'disease', 'disorder', 'condition', 'symptom', 'drug',
            'medication', 'procedure', 'surgery', 'intervention', 'outcome',
            'efficacy', 'safety', 'adverse', 'side effect', 'complication'
        ]

        words = text.lower().split()
        medical_count = sum(1 for word in words if any(term in word for term in medical_terms))

        return medical_count / len(words) if words else 0.0

    def get_preprocessing_stats(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> dict[str, Any]:
        """Obtener estadísticas del preprocesamiento"""
        return {
            'original_articles': len(df_original),
            'processed_articles': len(df_processed),
            'average_title_length_before': df_original['title'].str.len().mean(),
            'average_title_length_after': df_processed['title_clean'].str.len().mean(),
            'average_abstract_length_before': df_original['abstract'].str.len().mean(),
            'average_abstract_length_after': df_processed['abstract_clean'].str.len().mean(),
            'total_reduction_percent': (
                (df_original['title'].str.len().sum() + df_original['abstract'].str.len().sum() -
                 df_processed['title_clean'].str.len().sum() - df_processed['abstract_clean'].str.len().sum()) /
                (df_original['title'].str.len().sum() + df_original['abstract'].str.len().sum()) * 100
            )
        }
