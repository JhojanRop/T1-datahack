# main.py - API principal simplificada
import json
import traceback
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Importar las clases del notebook
from services.biobert_classifier_enhanced import BioBERTClassifierEnhanced
from services.hybrid_classifier_enhanced import HybridMedicalClassifierEnhanced
from services.llm_classifier_enhanced import MedicalLLMClassifier
from services.pipeline_enhanced import MedicalClassificationPipelineEnhanced
from utils.medical_evaluator import MedicalEvaluatorEnhanced
from utils.medical_preprocessor import MedicalTextPreprocessor

app = FastAPI(
    title="🏥 Medical Classification API",
    description="API para clasificación de literatura médica con BioBERT + LLM",
    version="2.0.0"
)

# CORS para la integración con v0
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para los modelos
biobert_model = None
llm_model = None
hybrid_model = None
pipeline = None
preprocessor = None

@app.on_event("startup")
async def startup_event():
    """Inicializar modelos al arrancar la API"""
    global biobert_model, llm_model, hybrid_model, pipeline, preprocessor

    try:
        print("🚀 Inicializando modelos médicos...")

        # Inicializar preprocesador
        preprocessor = MedicalTextPreprocessor()

        # Cargar BioBERT
        biobert_model = BioBERTClassifierEnhanced()
        biobert_model.load_model_from_local("model/biobert_finetuned_v3")

        # Configurar mapeo correcto de etiquetas
        true_label_mapping = {
            0: 'cardiovascular',
            1: 'hepatorenal',
            2: 'neurological',
            3: 'oncological'
        }
        biobert_model.label_names = [true_label_mapping[i] for i in range(4)]
        biobert_model.model.config.id2label = true_label_mapping

        # Inicializar LLM (modo simulado para demo)
        llm_model = MedicalLLMClassifier(api_key=None)  # Modo simulado

        # Crear sistema híbrido
        hybrid_model = HybridMedicalClassifierEnhanced(
            biobert_classifier=biobert_model,
            llm_classifier=llm_model,
            confidence_threshold=0.7
        )

        # Inicializar pipeline de producción
        pipeline = MedicalClassificationPipelineEnhanced(
            hybrid_classifier=hybrid_model,
            preprocessor=preprocessor,
            confidence_threshold=0.7
        )

        print("✅ Modelos inicializados correctamente")

    except Exception as e:
        print(f"❌ Error inicializando modelos: {e}")
        traceback.print_exc()

@app.get("/")
async def root():
    """Endpoint de salud de la API"""
    return {
        "message": "🏥 Medical Classification API v2.0",
        "status": "healthy",
        "models_loaded": {
            "biobert": biobert_model is not None,
            "llm": llm_model is not None,
            "hybrid": hybrid_model is not None,
            "pipeline": pipeline is not None
        }
    }

@app.post("/classify/upload-dataset")
async def upload_and_classify_dataset(file: UploadFile = File(...)):
    """
    Subir dataset CSV y realizar clasificación completa con todos los modelos
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")

    try:
        # Leer CSV
        content = await file.read()
        csv_content = StringIO(content.decode('utf-8'))
        df = pd.read_csv(csv_content, sep=';')  # Separador como en tu dataset

        # Validar estructura
        required_columns = ['title', 'abstract', 'group']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV debe contener las columnas: {required_columns}"
            )

        # Limitar dataset para demo (máximo 50 registros)
        if len(df) > 50:
            df = df.head(50)
            print("⚠️ Dataset limitado a 50 registros para demo")

        # Procesar dataset
        results = await process_dataset_with_all_models(df)

        return {
            "message": "Dataset procesado exitosamente",
            "dataset_info": {
                "total_articles": len(df),
                "columns": list(df.columns),
                "sample_titles": df['title'].head(3).tolist()
            },
            "results": results
        }

    except Exception as e:
        print(f"❌ Error procesando dataset: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando dataset: {str(e)}")

async def process_dataset_with_all_models(df: pd.DataFrame) -> dict[str, Any]:
    """Procesar dataset con todos los modelos y generar métricas"""

    # Preprocesar datos
    df_processed = preprocessor.preprocess_dataset(df)

    # Preparar etiquetas verdaderas
    from sklearn.preprocessing import MultiLabelBinarizer

    from utils.medical_label_analyzer import MedicalLabelAnalyzer

    label_analyzer = MedicalLabelAnalyzer()
    y_labels, mlb = label_analyzer.prepare_multilabel_targets(df_processed)

    # Preparar artículos para clasificación
    articles = [(row['title'], row['abstract']) for _, row in df_processed.iterrows()]

    results = {}

    # 1. Evaluar BioBERT
    print("🧬 Evaluando BioBERT...")
    biobert_results = await evaluate_biobert_model(articles, y_labels)
    results['biobert'] = biobert_results

    # 2. Evaluar LLM (máximo 10 casos)
    print("🤖 Evaluando LLM...")
    llm_articles = articles[:10]  # Limitar para demo
    llm_y_labels = y_labels.head(10) if len(articles) > 10 else y_labels
    llm_results = await evaluate_llm_model(llm_articles, llm_y_labels)
    results['llm'] = llm_results

    # 3. Evaluar Sistema Híbrido
    print("🔄 Evaluando Sistema Híbrido...")
    hybrid_results = await evaluate_hybrid_model(articles, y_labels)
    results['hybrid'] = hybrid_results

    # 4. Análisis de distribución de etiquetas
    print("📊 Analizando distribución de etiquetas...")
    label_analysis = analyze_label_distribution(df_processed)
    results['label_analysis'] = label_analysis

    return results

async def evaluate_biobert_model(articles: list[tuple], y_labels: pd.DataFrame) -> dict[str, Any]:
    """Evaluar modelo BioBERT"""
    try:
        # Obtener predicciones
        combined_texts = [f"{title} [SEP] {abstract}" for title, abstract in articles]
        biobert_predictions = biobert_model.predict_with_confidence_enhanced(
            combined_texts,
            confidence_threshold=0.7,
            confidence_method='difference'
        )

        # Convertir a formato de evaluación
        y_pred = []
        y_true = []

        for i, (title, abstract) in enumerate(articles):
            if i < len(biobert_predictions['all_predictions']):
                pred_probs = biobert_predictions['all_predictions'][i]
                pred_labels = [prob > 0.5 for prob in pred_probs]
                y_pred.append(pred_labels)

                true_labels = [y_labels.iloc[i][col] for col in biobert_model.label_names]
                y_true.append(true_labels)

        # Calcular métricas
        metrics = calculate_classification_metrics(np.array(y_true), np.array(y_pred))

        # Matriz de confusión
        confusion_matrices = calculate_confusion_matrices(np.array(y_true), np.array(y_pred), biobert_model.label_names)

        return {
            "model_name": "BioBERT Enhanced",
            "total_predictions": len(y_pred),
            "confidence_stats": {
                "mean_confidence": float(np.mean(biobert_predictions['all_confidence'])),
                "std_confidence": float(np.std(biobert_predictions['all_confidence'])),
                "min_confidence": float(np.min(biobert_predictions['all_confidence'])),
                "max_confidence": float(np.max(biobert_predictions['all_confidence']))
            },
            "metrics": metrics,
            "confusion_matrices": confusion_matrices,
            "processing_info": {
                "obvious_cases": len(biobert_predictions['obvious_cases']['indices']),
                "difficult_cases": len(biobert_predictions['difficult_cases']['indices'])
            }
        }

    except Exception as e:
        print(f"❌ Error evaluando BioBERT: {e}")
        return {"error": str(e)}

async def evaluate_llm_model(articles: list[tuple], y_labels: pd.DataFrame) -> dict[str, Any]:
    """Evaluar modelo LLM (limitado a 10 casos)"""
    try:
        # Obtener predicciones LLM
        llm_predictions = []
        for title, abstract in articles:
            pred = llm_model.classify_complex_case(title, abstract)
            llm_predictions.append(pred)

        # Convertir a formato de evaluación
        y_pred = []
        y_true = []

        for i, pred in enumerate(llm_predictions):
            pred_labels = [pred['classification'][label] for label in biobert_model.label_names]
            y_pred.append(pred_labels)

            true_labels = [y_labels.iloc[i][col] for col in biobert_model.label_names]
            y_true.append(true_labels)

        # Calcular métricas
        metrics = calculate_classification_metrics(np.array(y_true), np.array(y_pred))

        # Matriz de confusión
        confusion_matrices = calculate_confusion_matrices(np.array(y_true), np.array(y_pred), biobert_model.label_names)

        return {
            "model_name": "LLM (Gemini Simulated)",
            "total_predictions": len(y_pred),
            "average_confidence": float(np.mean([pred['confidence_score'] for pred in llm_predictions])),
            "metrics": metrics,
            "confusion_matrices": confusion_matrices,
            "note": "Limitado a 10 casos para demo"
        }

    except Exception as e:
        print(f"❌ Error evaluando LLM: {e}")
        return {"error": str(e)}

async def evaluate_hybrid_model(articles: list[tuple], y_labels: pd.DataFrame) -> dict[str, Any]:
    """Evaluar sistema híbrido"""
    try:
        # Obtener predicciones híbridas
        hybrid_predictions = hybrid_model.classify_batch(articles)

        # Convertir a formato de evaluación
        y_pred = []
        y_true = []

        for i, pred in enumerate(hybrid_predictions):
            pred_labels = [pred['classification'][label] for label in hybrid_model.label_names]
            y_pred.append(pred_labels)

            true_labels = [y_labels.iloc[i][col] for col in hybrid_model.label_names]
            y_true.append(true_labels)

        # Calcular métricas
        metrics = calculate_classification_metrics(np.array(y_true), np.array(y_pred))

        # Matriz de confusión
        confusion_matrices = calculate_confusion_matrices(np.array(y_true), np.array(y_pred), hybrid_model.label_names)

        # Estadísticas del sistema híbrido
        hybrid_stats = hybrid_model.get_performance_stats()

        return {
            "model_name": "Sistema Híbrido (BioBERT + LLM)",
            "total_predictions": len(y_pred),
            "metrics": metrics,
            "confusion_matrices": confusion_matrices,
            "hybrid_stats": hybrid_stats
        }

    except Exception as e:
        print(f"❌ Error evaluando sistema híbrido: {e}")
        return {"error": str(e)}

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcular métricas de clasificación multilabel"""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def calculate_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, list[list[int]]]:
    """Calcular matrices de confusión por cada etiqueta"""
    from sklearn.metrics import confusion_matrix

    confusion_matrices = {}

    for i, label in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        confusion_matrices[label] = cm.tolist()

    return confusion_matrices

def analyze_label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Analizar distribución de etiquetas"""
    from collections import Counter

    # Parsear etiquetas
    all_labels = []
    for group in df['group']:
        if pd.notna(group):
            labels = [label.strip() for label in str(group).split('|')]
            all_labels.extend(labels)

    label_counts = Counter(all_labels)
    return {
        "total_articles": len(df),
        "unique_labels": len(label_counts),
        "label_distribution": dict(label_counts),
        "most_common_labels": label_counts.most_common(5),
        "average_labels_per_article": len(all_labels) / len(df)
    }

@app.get("/models/info")
async def get_models_info():
    """Información sobre los modelos cargados"""
    return {
        "biobert": {
            "model_name": "BioBERT Enhanced",
            "status": "loaded" if biobert_model else "not_loaded",
            "label_names": biobert_model.label_names if biobert_model else [],
            "description": "Modelo BioBERT fine-tuned para clasificación médica"
        },
        "llm": {
            "model_name": "Gemini LLM (Simulated)",
            "status": "loaded" if llm_model else "not_loaded",
            "description": "LLM para casos complejos (modo simulado para demo)"
        },
        "hybrid": {
            "model_name": "Sistema Híbrido",
            "status": "loaded" if hybrid_model else "not_loaded",
            "confidence_threshold": hybrid_model.confidence_threshold if hybrid_model else None,
            "description": "Combinación inteligente de BioBERT + LLM"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
