import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import os


# Carga de WiC
def cargar_wic(ruta_data, ruta_gold):
    print("Cargando dataset WiC.")
    dataset = []
    with open(ruta_data, 'r', encoding='utf-8') as fd, open(ruta_gold, 'r', encoding='utf-8') as fg:
        for linea_data, linea_gold in zip(fd, fg):
            partes = linea_data.strip().split('\t')
            if len(partes) < 5: continue
                
            palabra, pos, indices, sent1, sent2 = partes
            idx1, idx2 = map(int, indices.split('-'))
            label = 1 if linea_gold.strip() == 'T' else 0
            
            dataset.append({
                "word": palabra,
                "idx1": idx1,
                "idx2": idx2,
                "sent1": sent1.split(), 
                "sent2": sent2.split(),
                "label": label
            })
    print(f"Cargados {len(dataset)} pares de oraciones.")
    return dataset

# Extracción de palabra
def obtener_embedding_palabra(hidden_states, word_ids, target_word_idx):
    indices_tokens = [i for i, w_id in enumerate(word_ids) if w_id == target_word_idx]
    emb_por_capa = []
    
    for capa in hidden_states:
        vectores_subtokens = capa[0, indices_tokens, :]
        vector_palabra = torch.mean(vectores_subtokens, dim=0)
        emb_por_capa.append(vector_palabra.cpu().numpy().astype(np.float16))
        
    return emb_por_capa


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo detectado: {device}")


model_name = "citiusLTL/AnorBERT"
print(f"Cargando modelo: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
model.to(device)
model.eval()

ruta_data = "wic_data.txt" 
ruta_gold = "wic_gold.txt"
dataset = cargar_wic(ruta_data, ruta_gold)

X_features = {i: [] for i in range(13)}
y_labels = []

print("\nIniciando extracción de características.")
# Procesamos el dataset
for idx, item in enumerate(dataset):
    y_labels.append(item["label"])
    
    # Frase 1
    inputs1 = tokenizer(item["sent1"], is_split_into_words=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out1 = model(**inputs1)
    emb_palabra_1 = obtener_embedding_palabra(out1.hidden_states, inputs1.word_ids(), item["idx1"])
    
    # Frase 2
    inputs2 = tokenizer(item["sent2"], is_split_into_words=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model(**inputs2)
    emb_palabra_2 = obtener_embedding_palabra(out2.hidden_states, inputs2.word_ids(), item["idx2"])
    
    # Calculamos la relación y guardamos por capa
    for capa in range(13):
        e1 = emb_palabra_1[capa]
        e2 = emb_palabra_2[capa]
        feature_vector = np.concatenate([e1, e2, np.abs(e1 - e2), e1 * e2])
        X_features[capa].append(feature_vector)
        
    if (idx + 1) % 500 == 0:
        print(f"Procesadas {idx + 1} / {len(dataset)} oraciones.")


print("\nExtracción finalizada. Guardando matrices en disco.")

output_dir = "logs_wic"
os.makedirs(output_dir, exist_ok=True)

# Guardar etiquetas
np.save(os.path.join(output_dir, "wic_labels.npy"), np.array(y_labels))

# Guardar capas
for capa in range(13):
    archivo_capa = os.path.join(output_dir, f"wic_features_capa_{capa}.npy")
    matriz_capa = np.array(X_features[capa], dtype=np.float16) 
    np.save(archivo_capa, matriz_capa)
    print(f"Guardada Capa {capa:02d} -> Forma: {matriz_capa.shape}")

print(f"\nTodo guardado en la carpeta '{output_dir}'.")