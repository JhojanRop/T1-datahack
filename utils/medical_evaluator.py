from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)


class MedicalEvaluatorEnhanced:
    """Evaluador mejorado para clasificación médica multilabel"""

    def __init__(self, label_names: list[str]):
        self.label_names = label_names
        self.n_labels = len(label_names)
        print(f"📊 Evaluador médico inicializado para {self.n_labels} etiquetas: {label_names}")

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: np.ndarray = None) -> dict[str, Any]:
        """Evaluación completa de predicciones multilabel"""

        print("🔍 Realizando evaluación completa...")

        results = {
            'basic_metrics': self._calculate_basic_metrics(y_true, y_pred),
            'multilabel_metrics': self._calculate_multilabel_metrics(y_true, y_pred),
            'per_label_metrics': self._calculate_per_label_metrics(y_true, y_pred),
            'confusion_matrices': self._calculate_confusion_matrices(y_true, y_pred),
            'classification_report': self._generate_classification_report(y_true, y_pred)
        }

        if y_pred_proba is not None:
            results['probability_metrics'] = self._calculate_probability_metrics(y_true, y_pred_proba)

        print("✅ Evaluación completada")
        return results

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Métricas básicas de clasificación"""
        return {
            'exact_match_ratio': accuracy_score(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
            'subset_accuracy': accuracy_score(y_true, y_pred)
        }

    def _calculate_multilabel_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Métricas específicas para multilabel"""
        return {
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_samples': precision_score(y_true, y_pred, average='samples', zero_division=0),

            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_samples': recall_score(y_true, y_pred, average='samples', zero_division=0),

            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0)
        }

    def _calculate_per_label_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
        """Métricas por cada etiqueta individual"""
        per_label = {}

        for i, label_name in enumerate(self.label_names):
            per_label[label_name] = {
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'f1_score': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'support': np.sum(y_true[:, i]),
                'predicted_positive': np.sum(y_pred[:, i]),
                'true_positive': np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)),
                'false_positive': np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)),
                'false_negative': np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
            }

        return per_label

    def _calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, list[list[int]]]:
        """Matrices de confusión por etiqueta"""
        confusion_matrices = {}

        for i, label_name in enumerate(self.label_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[label_name] = cm.tolist()

        return confusion_matrices

    def _generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Reporte de clasificación detallado"""
        reports = {}

        for i, label_name in enumerate(self.label_names):
            report = classification_report(
                y_true[:, i], y_pred[:, i],
                output_dict=True,
                zero_division=0
            )
            reports[label_name] = report

        return reports

    def _calculate_probability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, Any]:
        """Métricas basadas en probabilidades"""
        from sklearn.metrics import average_precision_score, roc_auc_score

        prob_metrics = {}

        for i, label_name in enumerate(self.label_names):
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])

                # Average Precision
                avg_precision = average_precision_score(y_true[:, i], y_pred_proba[:, i])

                prob_metrics[label_name] = {
                    'roc_auc': roc_auc,
                    'average_precision': avg_precision,
                    'mean_predicted_prob': np.mean(y_pred_proba[:, i]),
                    'std_predicted_prob': np.std(y_pred_proba[:, i])
                }
            except ValueError as e:
                # Manejar casos donde no hay ejemplos positivos
                prob_metrics[label_name] = {
                    'roc_auc': 0.5,
                    'average_precision': 0.0,
                    'mean_predicted_prob': np.mean(y_pred_proba[:, i]),
                    'std_predicted_prob': np.std(y_pred_proba[:, i]),
                    'error': str(e)
                }

        return prob_metrics

    def calculate_medical_insights(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 texts: list[str] = None) -> dict[str, Any]:
        """Análisis médico especializado"""

        insights = {
            'label_distribution': self._analyze_label_distribution(y_true, y_pred),
            'prediction_patterns': self._analyze_prediction_patterns(y_true, y_pred),
            'medical_complexity': self._analyze_medical_complexity(y_true, y_pred)
        }

        if texts:
            insights['text_analysis'] = self._analyze_prediction_by_text_features(y_true, y_pred, texts)

        return insights

    def _analyze_label_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Análisis de distribución de etiquetas"""
        return {
            'true_distribution': {
                label: int(np.sum(y_true[:, i]))
                for i, label in enumerate(self.label_names)
            },
            'predicted_distribution': {
                label: int(np.sum(y_pred[:, i]))
                for i, label in enumerate(self.label_names)
            },
            'distribution_shift': {
                label: int(np.sum(y_pred[:, i]) - np.sum(y_true[:, i]))
                for i, label in enumerate(self.label_names)
            }
        }

    def _analyze_prediction_patterns(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Análisis de patrones de predicción"""

        # Combinaciones de etiquetas más comunes
        true_combinations = {}
        pred_combinations = {}

        for i in range(len(y_true)):
            true_combo = tuple(j for j, val in enumerate(y_true[i]) if val == 1)
            pred_combo = tuple(j for j, val in enumerate(y_pred[i]) if val == 1)

            true_combinations[true_combo] = true_combinations.get(true_combo, 0) + 1
            pred_combinations[pred_combo] = pred_combinations.get(pred_combo, 0) + 1

        return {
            'most_common_true_combinations': sorted(true_combinations.items(), key=lambda x: x[1], reverse=True)[:5],
            'most_common_predicted_combinations': sorted(pred_combinations.items(), key=lambda x: x[1], reverse=True)[:5],
            'single_label_cases': np.sum(np.sum(y_true, axis=1) == 1),
            'multi_label_cases': np.sum(np.sum(y_true, axis=1) > 1),
            'no_label_cases': np.sum(np.sum(y_true, axis=1) == 0)
        }

    def _analyze_medical_complexity(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Análisis de complejidad médica"""

        # Casos por número de etiquetas
        true_label_counts = np.sum(y_true, axis=1)
        pred_label_counts = np.sum(y_pred, axis=1)

        complexity_analysis = {}
        for n_labels in range(0, self.n_labels + 1):
            true_cases = np.sum(true_label_counts == n_labels)
            pred_cases = np.sum(pred_label_counts == n_labels)

            if true_cases > 0:
                # Accuracy para casos con n etiquetas
                mask = true_label_counts == n_labels
                accuracy = accuracy_score(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else 0

                complexity_analysis[f'{n_labels}_labels'] = {
                    'true_cases': int(true_cases),
                    'predicted_cases': int(pred_cases),
                    'accuracy': float(accuracy)
                }

        return complexity_analysis

    def _analyze_prediction_by_text_features(self, y_true: np.ndarray, y_pred: np.ndarray,
                                           texts: list[str]) -> dict[str, Any]:
        """Análisis de predicciones por características del texto"""

        text_lengths = [len(text.split()) for text in texts]

        # Dividir en cuartiles por longitud
        length_quartiles = np.percentile(text_lengths, [25, 50, 75])

        analysis = {}
        quartile_names = ['short', 'medium', 'long', 'very_long']

        for i, quartile_name in enumerate(quartile_names):
            if i == 0:
                mask = np.array(text_lengths) <= length_quartiles[0]
            elif i == 1:
                mask = (np.array(text_lengths) > length_quartiles[0]) & (np.array(text_lengths) <= length_quartiles[1])
            elif i == 2:
                mask = (np.array(text_lengths) > length_quartiles[1]) & (np.array(text_lengths) <= length_quartiles[2])
            else:
                mask = np.array(text_lengths) > length_quartiles[2]

            if np.sum(mask) > 0:
                accuracy = accuracy_score(y_true[mask], y_pred[mask])
                analysis[quartile_name] = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(accuracy),
                    'avg_length': float(np.mean(np.array(text_lengths)[mask]))
                }

        return analysis

    def generate_evaluation_summary(self, evaluation_results: dict[str, Any]) -> str:
        """Generar resumen textual de la evaluación"""

        basic = evaluation_results['basic_metrics']
        multilabel = evaluation_results['multilabel_metrics']

        summary = f"""
📊 RESUMEN DE EVALUACIÓN MÉDICA
{'='*50}

📈 MÉTRICAS PRINCIPALES:
• Accuracy Exacta: {basic['exact_match_ratio']:.3f}
• F1-Score Macro: {multilabel['f1_macro']:.3f}
• F1-Score Micro: {multilabel['f1_micro']:.3f}
• F1-Score Weighted: {multilabel['f1_weighted']:.3f}

🎯 PRECISIÓN Y RECALL:
• Precisión Macro: {multilabel['precision_macro']:.3f}
• Recall Macro: {multilabel['recall_macro']:.3f}
• Hamming Loss: {basic['hamming_loss']:.3f}

📋 RENDIMIENTO POR ETIQUETA:
"""

        per_label = evaluation_results['per_label_metrics']
        for label, metrics in per_label.items():
            summary += f"• {label.capitalize()}: F1={metrics['f1_score']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}\n"

        return summary
