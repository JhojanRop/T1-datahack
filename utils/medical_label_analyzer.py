from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class MedicalLabelAnalyzer:
    """Analizador de etiquetas médicas para problemas multilabel"""

    def __init__(self):
        self.mlb = None
        self.label_names = None
        self.label_stats = {}
        print("🏷️ Analizador de etiquetas médicas inicializado")

    def prepare_multilabel_targets(self, df: pd.DataFrame, label_column: str = 'group') -> tuple[pd.DataFrame, MultiLabelBinarizer]:
        """Preparar targets multilabel desde el dataset"""

        print(f"🔄 Procesando etiquetas de columna '{label_column}'...")

        # Procesar etiquetas
        labels_list = []
        for idx, row in df.iterrows():
            labels = self._parse_labels(row[label_column])
            labels_list.append(labels)

        # Crear MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        y_binary = self.mlb.fit_transform(labels_list)

        # Guardar nombres de etiquetas
        self.label_names = list(self.mlb.classes_)

        # Crear DataFrame con etiquetas binarias
        y_df = pd.DataFrame(y_binary, columns=self.label_names, index=df.index)

        # Calcular estadísticas
        self._calculate_label_statistics(y_df, labels_list)

        print(f"✅ Procesadas {len(self.label_names)} etiquetas únicas:")
        print(f"   📊 Etiquetas: {self.label_names}")
        print(f"   📈 Distribución: {dict(self.label_stats['frequency'])}")

        return y_df, self.mlb

    def _parse_labels(self, label_string: str) -> list[str]:
        """Parsear string de etiquetas separadas por |"""
        if pd.isna(label_string) or not isinstance(label_string, str):
            return []

        # Separar por | y limpiar
        labels = [label.strip().lower() for label in str(label_string).split('|')]

        # Filtrar etiquetas vacías
        labels = [label for label in labels if label]

        # Normalizar etiquetas conocidas
        normalized_labels = []
        for label in labels:
            normalized = self._normalize_label(label)
            if normalized and normalized not in normalized_labels:
                normalized_labels.append(normalized)

        return normalized_labels

    def _normalize_label(self, label: str) -> str:
        """Normalizar nombres de etiquetas"""

        # Mapeo de normalizaciones
        normalizations = {
            # Cardiovascular
            'cardio': 'cardiovascular',
            'cardiac': 'cardiovascular',
            'heart': 'cardiovascular',
            'vascular': 'cardiovascular',

            # Neurológico
            'neuro': 'neurological',
            'neural': 'neurological',
            'brain': 'neurological',
            'cerebral': 'neurological',

            # Oncológico
            'onco': 'oncological',
            'cancer': 'oncological',
            'tumor': 'oncological',
            'malignant': 'oncological',

            # Hepatorenal
            'hepato': 'hepatorenal',
            'renal': 'hepatorenal',
            'kidney': 'hepatorenal',
            'liver': 'hepatorenal',
            'hepatic': 'hepatorenal'
        }

        return normalizations.get(label, label)

    def _calculate_label_statistics(self, y_df: pd.DataFrame, labels_list: list[list[str]]):
        """Calcular estadísticas de las etiquetas"""

        # Frecuencia por etiqueta
        label_counts = y_df.sum().sort_values(ascending=False)

        # Combinaciones de etiquetas
        combination_counts = Counter([tuple(sorted(labels)) for labels in labels_list])

        # Estadísticas por artículo
        labels_per_article = [len(labels) for labels in labels_list]

        self.label_stats = {
            'frequency': label_counts.to_dict(),
            'combinations': dict(combination_counts.most_common(10)),
            'articles_per_label': {
                label: int(count) for label, count in label_counts.items()
            },
            'label_distribution': {
                'mean_labels_per_article': np.mean(labels_per_article),
                'std_labels_per_article': np.std(labels_per_article),
                'max_labels_per_article': max(labels_per_article),
                'min_labels_per_article': min(labels_per_article),
                'single_label_articles': sum(1 for x in labels_per_article if x == 1),
                'multi_label_articles': sum(1 for x in labels_per_article if x > 1),
                'no_label_articles': sum(1 for x in labels_per_article if x == 0)
            }
        }

    def analyze_label_correlations(self, y_df: pd.DataFrame) -> dict[str, Any]:
        """Analizar correlaciones entre etiquetas"""

        # Matriz de correlación
        correlation_matrix = y_df.corr()

        # Co-ocurrencias
        cooccurrence_matrix = np.dot(y_df.T.values, y_df.values)
        cooccurrence_df = pd.DataFrame(
            cooccurrence_matrix,
            index=y_df.columns,
            columns=y_df.columns
        )

        # Pares más correlacionados
        correlations = []
        for i in range(len(self.label_names)):
            for j in range(i+1, len(self.label_names)):
                label1, label2 = self.label_names[i], self.label_names[j]
                corr = correlation_matrix.loc[label1, label2]
                cooc = cooccurrence_df.loc[label1, label2]

                correlations.append({
                    'label1': label1,
                    'label2': label2,
                    'correlation': corr,
                    'cooccurrence': int(cooc),
                    'jaccard_similarity': cooc / (y_df[label1].sum() + y_df[label2].sum() - cooc)
                })

        # Ordenar por correlación
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'cooccurrence_matrix': cooccurrence_df.to_dict(),
            'top_correlations': correlations[:10],
            'highly_correlated_pairs': [
                (c['label1'], c['label2'], c['correlation'])
                for c in correlations[:5] if abs(c['correlation']) > 0.3
            ]
        }

    def analyze_label_complexity(self, y_df: pd.DataFrame) -> dict[str, Any]:
        """Analizar complejidad de las etiquetas"""

        # Análisis por número de etiquetas por artículo
        labels_per_article = y_df.sum(axis=1)

        complexity_analysis = {}
        for n_labels in range(0, len(self.label_names) + 1):
            articles_with_n_labels = (labels_per_article == n_labels).sum()
            if articles_with_n_labels > 0:
                complexity_analysis[f'{n_labels}_labels'] = {
                    'count': int(articles_with_n_labels),
                    'percentage': float(articles_with_n_labels / len(y_df) * 100)
                }

        # Etiquetas más difíciles (que aparecen con muchas otras)
        label_complexity = {}
        for label in self.label_names:
            mask = y_df[label] == 1
            if mask.sum() > 0:
                avg_other_labels = y_df[mask].drop(columns=[label]).sum(axis=1).mean()
                label_complexity[label] = {
                    'avg_cooccurring_labels': float(avg_other_labels),
                    'frequency': int(mask.sum()),
                    'complexity_score': float(avg_other_labels * mask.sum())
                }

        return {
            'distribution_by_label_count': complexity_analysis,
            'label_complexity_scores': label_complexity,
            'most_complex_labels': sorted(
                label_complexity.items(),
                key=lambda x: x[1]['complexity_score'],
                reverse=True
            )[:5]
        }

    def get_imbalance_metrics(self, y_df: pd.DataFrame) -> dict[str, Any]:
        """Calcular métricas de desbalance de clases"""

        total_samples = len(y_df)
        imbalance_metrics = {}

        for label in self.label_names:
            positive_samples = y_df[label].sum()
            negative_samples = total_samples - positive_samples

            # Ratio de desbalance
            imbalance_ratio = negative_samples / positive_samples if positive_samples > 0 else float('inf')

            # Frecuencia relativa
            positive_rate = positive_samples / total_samples

            imbalance_metrics[label] = {
                'positive_samples': int(positive_samples),
                'negative_samples': int(negative_samples),
                'imbalance_ratio': float(imbalance_ratio),
                'positive_rate': float(positive_rate),
                'severity': self._classify_imbalance_severity(imbalance_ratio)
            }

        return imbalance_metrics

    def _classify_imbalance_severity(self, ratio: float) -> str:
        """Clasificar severidad del desbalance"""
        if ratio < 2:
            return 'balanced'
        elif ratio < 5:
            return 'mild_imbalance'
        elif ratio < 20:
            return 'moderate_imbalance'
        elif ratio < 100:
            return 'severe_imbalance'
        else:
            return 'extreme_imbalance'

    def generate_analysis_report(self) -> str:
        """Generar reporte de análisis de etiquetas"""

        if not self.label_stats:
            return "❌ No hay estadísticas disponibles. Ejecute prepare_multilabel_targets primero."

        report = """
🏷️ ANÁLISIS DE ETIQUETAS MÉDICAS
{'='*50}

📊 DISTRIBUCIÓN GENERAL:
• Total de etiquetas únicas: {len(self.label_names)}
• Etiquetas por artículo (promedio): {self.label_stats['label_distribution']['mean_labels_per_article']:.2f}
• Artículos con una sola etiqueta: {self.label_stats['label_distribution']['single_label_articles']}
• Artículos con múltiples etiquetas: {self.label_stats['label_distribution']['multi_label_articles']}

📈 FRECUENCIA POR ETIQUETA:
"""

        for label, count in sorted(self.label_stats['frequency'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(self.label_stats['frequency'].values())) * 100
            report += f"• {label.capitalize()}: {count} artículos ({percentage:.1f}%)\n"

        report += """
🔗 COMBINACIONES MÁS COMUNES:
"""

        for combo, count in list(self.label_stats['combinations'].items())[:5]:
            combo_str = " + ".join(combo) if combo else "Sin etiquetas"
            report += f"• {combo_str}: {count} artículos\n"

        return report
