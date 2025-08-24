# ==============================================================================
# 1. IMPORTAR LIBRERÍAS
# ==============================================================================
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import torch # <<< NUEVO: Necesario para el modelo BERT
from transformers import BertTokenizer, BertModel # <<< NUEVO: Para generar embeddings

print("--- Inicio del script de generación de datos para V0 ---")

# ==============================================================================
# 2. CONFIGURACIÓN DE RUTAS Y MODELOS
# ==============================================================================
CLASSIFIER_MODEL_PATH = 'models/mlp_sigmiod_per_class.h5'
DATA_PATH = 'data/dataset.csv'
V0_OUTPUT_DIR = 'v0_data'
BERT_MODEL_NAME = 'bert-base-uncased' # <<< NUEVO: Modelo para embeddings

# ==============================================================================
# 3. PREPARACIÓN DEL ENTORNO Y MODELOS DE EMBEDDING
# ==============================================================================
os.makedirs(V0_OUTPUT_DIR, exist_ok=True)
print(f"Directorio '{V0_OUTPUT_DIR}' asegurado.")

# Configurar dispositivo para PyTorch (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo para PyTorch: {device}")

# Cargar tokenizer y modelo BERT para embeddings
print(f"Cargando tokenizer y modelo BERT: '{BERT_MODEL_NAME}'...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
embedding_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)

# ==============================================================================
# 4. CARGA Y PREPROCESAMIENTO DE DATOS
# ==============================================================================
print("Cargando y preprocesando datos...")
try:
    # <<< CAMBIO: Usamos sep=';' para leer el CSV correctamente
    df = pd.read_csv(DATA_PATH, sep=';')
    print(f"Datos cargados. {len(df)} registros encontrados.")
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de datos en la ruta: {DATA_PATH}")
    exit()

# Combinar título y abstract para formar el texto de entrada
df['text'] = (df['title'].fillna("") + " " + df['abstract'].fillna("")).str.strip()

# Preparar etiquetas (y_true) usando MultiLabelBinarizer
target_columns = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
y_lists = [s.split('|') if isinstance(s, str) and s else [] for s in df["group"]]
mlb = MultiLabelBinarizer(classes=target_columns)
y_true = mlb.fit_transform(y_lists)

# ==============================================================================
# 5. FUNCIÓN PARA GENERAR EMBEDDINGS (Vectorización)
# ==============================================================================
def get_mean_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        token_embeddings = model.embeddings.word_embeddings(inputs["input_ids"])
        token_matrix = token_embeddings.squeeze(0)
        mean_embedding = token_matrix.mean(dim=0)
    return mean_embedding.cpu().numpy()

# ==============================================================================
# 6. GENERAR EMBEDDINGS PARA TODO EL DATASET
# ==============================================================================
print("Generando embeddings de BERT para los textos...")
# Esto puede tardar unos minutos si tienes muchos datos. Se recomienda hacer en batches.
X_embeddings = np.vstack([get_mean_embedding(text, tokenizer, embedding_model, device) for text in df["text"].values])
print(f"Embeddings generados. Shape de la matriz X: {X_embeddings.shape}")

# ==============================================================================
# 7. CARGAR MODELO CLASIFICADOR Y REALIZAR PREDICCIONES
# ==============================================================================
print("Cargando modelo clasificador Keras...")
try:
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
except Exception as e:
    print(f"ERROR al cargar el modelo Keras: {e}")
    exit()

print("Realizando predicciones...")
y_pred_proba = classifier_model.predict(X_embeddings)
# Convertir probabilidades a etiquetas binarias (0 o 1)
y_pred = (y_pred_proba > 0.5).astype(int)

# ==============================================================================
# 8. GENERAR Y GUARDAR ARCHIVOS PARA V0 (Sin cambios aquí)
# ==============================================================================
print("Generando reporte de clasificación...")
report = classification_report(y_true, y_pred, target_names=target_columns, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'class'})
metrics_output_file = os.path.join(V0_OUTPUT_DIR, 'classification_metrics.csv')
df_report.to_csv(metrics_output_file, index=False)
print(f"Métricas de clasificación guardadas en: {metrics_output_file}")

print("Generando datos para la matriz de confusión...")
mcm = multilabel_confusion_matrix(y_true, y_pred)
confusion_data = []
for i, label in enumerate(target_columns):
    tn, fp, fn, tp = mcm[i].ravel()
    confusion_data.append({'class': label, 'metric': 'True Negative', 'value': tn})
    confusion_data.append({'class': label, 'metric': 'False Positive', 'value': fp})
    confusion_data.append({'class': label, 'metric': 'False Negative', 'value': fn})
    confusion_data.append({'class': label, 'metric': 'True Positive', 'value': tp})

df_confusion = pd.DataFrame(confusion_data)
confusion_output_file = os.path.join(V0_OUTPUT_DIR, 'confusion_matrix_data.csv')
df_confusion.to_csv(confusion_output_file, index=False)
print(f"Datos de matriz de confusión guardados en: {confusion_output_file}")

# ... (todo tu código hasta aquí)

df_confusion = pd.DataFrame(confusion_data)
confusion_output_file = os.path.join(V0_OUTPUT_DIR, 'confusion_matrix_data.csv')
df_confusion.to_csv(confusion_output_file, index=False)
print(f"Datos de matriz de confusión guardados en: {confusion_output_file}")

# ==============================================================================
# 9. <<< NUEVA SECCIÓN: GENERAR Y GUARDAR ARCHIVOS JSON PARA DASHBOARD
# ==============================================================================
print("\nGenerando archivos JSON para el dashboard de v0...")

# 9.1. Convertir el DataFrame de métricas a JSON
metrics_json_output_file = os.path.join(V0_OUTPUT_DIR, 'classification_metrics.json')
# 'orient="records"' crea una lista de objetos, ideal para JavaScript.
# 'indent=4' hace que el archivo sea legible para los humanos.
df_report.to_json(metrics_json_output_file, orient='records', indent=4)
print(f"Métricas en formato JSON guardadas en: {metrics_json_output_file}")

# 9.2. Convertir el DataFrame de la matriz de confusión a JSON
confusion_json_output_file = os.path.join(V0_OUTPUT_DIR, 'confusion_matrix_data.json')
df_confusion.to_json(confusion_json_output_file, orient='records', indent=4)
print(f"Datos de matriz de confusión en JSON guardados en: {confusion_json_output_file}")


print("\n--- Script finalizado exitosamente. ---") 
