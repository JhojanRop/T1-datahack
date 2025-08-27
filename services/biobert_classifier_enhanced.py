# services/biobert_classifier_enhanced.py
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


class BioBERTClassifierEnhanced:
    """Clasificador BioBERT mejorado extraído del notebook"""

    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.label_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model_from_local(self, model_path: str):
        """Cargar modelo BioBERT fine-tuned"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.is_trained = True
            print(f"✅ Modelo BioBERT cargado desde: {model_path}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def predict_with_confidence_enhanced(self, texts, confidence_threshold=0.7, confidence_method='difference'):
        """Realizar predicciones con scores de confianza"""
        if not self.is_trained:
            raise ValueError("❌ Modelo no entrenado")

        # Tokenizar
        tokenized_data = self.tokenize_data(texts)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(tokenized_data, batch_size=16, collate_fn=data_collator)

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}
                outputs = self.model(**inputs)
                predictions = outputs.logits.cpu().numpy()
                all_predictions.append(predictions)

        all_predictions = np.vstack(all_predictions)
        confidence_scores, probabilities = self.calculate_confidence_scores_robust(all_predictions, confidence_method)

        # Separar casos obvios vs difíciles
        obvious_mask = confidence_scores >= confidence_threshold
        difficult_mask = ~obvious_mask

        return {
            'obvious_cases': {
                'indices': np.where(obvious_mask)[0],
                'predictions': probabilities[obvious_mask],
                'confidence_scores': confidence_scores[obvious_mask],
                'texts': [texts[i] for i in np.where(obvious_mask)[0]]
            },
            'difficult_cases': {
                'indices': np.where(difficult_mask)[0],
                'texts': [texts[i] for i in np.where(difficult_mask)[0]],
                'confidence_scores': confidence_scores[difficult_mask]
            },
            'all_predictions': probabilities,
            'all_confidence': confidence_scores,
            'confidence_method': confidence_method
        }

    def tokenize_data(self, texts):
        """Tokenizar textos para BioBERT"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        return tokenized_dataset

    def calculate_confidence_scores_robust(self, predictions, method='difference'):
        """Calcular scores de confianza robustos"""
        # Aplicar sigmoid para obtener probabilidades
        probabilities = 1 / (1 + np.exp(-predictions))

        if method == 'difference':
            max_probs = np.max(probabilities, axis=1)
            second_max_probs = np.partition(probabilities, -2, axis=1)[:, -2]
            confidence_scores = max_probs - second_max_probs
        elif method == 'max_prob':
            confidence_scores = np.max(probabilities, axis=1)
        else:
            confidence_scores = np.max(probabilities, axis=1)

        return confidence_scores, probabilities
