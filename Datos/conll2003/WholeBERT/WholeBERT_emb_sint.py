import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel


# Función para leer CoNLL2003
def cargar_conll_local(ruta_archivo):
    print(f"Leyendo dataset desde: {ruta_archivo}")
    oraciones = []
    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []

    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            
            # Fin de oración
            if linea == "" or linea.startswith("-DOCSTART-"):
                if len(tokens) > 0:
                    oraciones.append({
                        "tokens": tokens,
                        "pos_tags": pos_tags,
                        "chunk_tags": chunk_tags,
                        "ner_tags": ner_tags
                    })
                    tokens, pos_tags, chunk_tags, ner_tags = [], [], [], []
            else:
                partes = linea.split()
                if len(partes) == 4:
                    tokens.append(partes[0])
                    pos_tags.append(partes[1])
                    chunk_tags.append(partes[2])
                    ner_tags.append(partes[3])

        # Última oración
        if len(tokens) > 0:
            oraciones.append({
                "tokens": tokens, "pos_tags": pos_tags, 
                "chunk_tags": chunk_tags, "ner_tags": ner_tags
            })
            
    print(f"¡Carga completada! Se encontraron {len(oraciones)} oraciones.")
    return oraciones


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Hardware detectado: {device}")

model_name = "citiusLTL/WholeBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
model.to(device)
model.eval()

# Carga de datos
dataset = cargar_conll_local("train_conll2003.txt")

output_dir = "logs_emb"
os.makedirs(output_dir, exist_ok=True)

print("Iniciando extracción de embeddings a nivel de token por bloques.")

TAMANO_BLOQUE = 500
bloque_actual = []
num_bloque = 0

# Extracción de embeddings por bloques

for idx, item in enumerate(dataset):
    words = item["tokens"] 
    pos_tags = item["pos_tags"]
    chunk_tags = item["chunk_tags"]
    ner_tags = item["ner_tags"]
    
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states
    word_ids = inputs.word_ids(batch_index=0)
    
    oracion_embeddings = {f"capa_{i}": [] for i in range(13)}
    palabras_procesadas = set()
    indices_validos = []
    
    # Filtrar para quedarse solo con la primera sub-palabra (token)
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in palabras_procesadas:
            indices_validos.append(token_idx)
            palabras_procesadas.add(word_id)
    
    # Extraer capas y convertirlas a float16
    for layer_idx, layer_tensor in enumerate(hidden_states):
        layer_np = layer_tensor.squeeze(0).cpu().numpy().astype(np.float16)
        embeddings_filtrados = layer_np[indices_validos]
        oracion_embeddings[f"capa_{layer_idx}"] = embeddings_filtrados
        
    datos_oracion = {
        "id_oracion": idx,
        "pos_labels": pos_tags,
        "chunk_labels": chunk_tags,
        "ner_labels": ner_tags,
        "embeddings": oracion_embeddings
    }
    bloque_actual.append(datos_oracion)

    # Guardado por bloques
    if len(bloque_actual) >= TAMANO_BLOQUE:
        archivo_salida = os.path.join(output_dir, f"conll_bloque_{num_bloque}.npy")
        np.save(archivo_salida, np.array(bloque_actual, dtype=object))
        print(f"Bloque {num_bloque} guardado con {len(bloque_actual)} oraciones.")
        bloque_actual = []
        num_bloque += 1

# Guardar lo que sobre al final
if len(bloque_actual) > 0:
    archivo_salida = os.path.join(output_dir, f"conll_bloque_{num_bloque}.npy")
    np.save(archivo_salida, np.array(bloque_actual, dtype=object))
    print(f"Bloque final {num_bloque} guardado con {len(bloque_actual)} oraciones.")

print(f"Extracción completada.")