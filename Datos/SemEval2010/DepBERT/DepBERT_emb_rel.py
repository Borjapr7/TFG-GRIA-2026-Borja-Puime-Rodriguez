import torch
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoModel

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Función para parsear el formato específico de SemEval-2010 Task 8
def cargar_semeval_local(ruta_archivo):
    print(f"Leyendo dataset SemEval desde: {ruta_archivo}")
    dataset = []
    
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    i = 0
    while i < len(lineas):
        linea = lineas[i].strip()
        
        # Ignorar líneas en blanco
        if not linea:
            i += 1
            continue
            
        # ID y Oración
        partes = linea.split('\t')
        if len(partes) < 2:
            i += 1
            continue
            
        id_oracion = partes[0]
        texto_crudo = partes[1].strip('"')
        
        # Limpiamos las etiquetas XML
        texto_limpio = re.sub(r'</?e[12]>', '', texto_crudo)
        
        # La relación
        i += 1
        relacion = lineas[i].strip()
        
        # Comentario (Lo ignoramos)
        i += 1
        
        # Guardamos el ejemplo
        dataset.append({
            "id": id_oracion,
            "texto": texto_limpio,
            "relacion": relacion
        })
        
        i += 1 # Avanzar a la siguiente posible línea en blanco
        
    print(f"¡Carga completada! Se encontraron {len(dataset)} relaciones.")
    return dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Hardware detectado: {device}")

model_name = "citiusLTL/DepBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
model.to(device)
model.eval()

# Carga de datos
archivo_semeval = "SemEval_Train.TXT" 
dataset = cargar_semeval_local(archivo_semeval)

output_dir = "logs_relacional"
os.makedirs(output_dir, exist_ok=True)

print("Iniciando extracción de embeddings del token [CLS] por bloques.")

TAMANO_BLOQUE = 500
bloque_actual = []
num_bloque = 0

# Extracción por bloques

for idx, item in enumerate(dataset):
    texto = item["texto"] 
    relacion = item["relacion"]
    
    # Tokenización normal de la oración completa
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states
    
    oracion_embeddings = {f"capa_{i}": [] for i in range(13)}
    
    # Nos quedamos con el token [CLS]
    for layer_idx, layer_tensor in enumerate(hidden_states):
        cls_embedding = layer_tensor[0, 0, :].cpu().numpy().astype(np.float16)
        oracion_embeddings[f"capa_{layer_idx}"] = cls_embedding
        
    datos_oracion = {
        "id_oracion": item["id"],
        "relation_label": relacion,
        "embeddings": oracion_embeddings
    }
    bloque_actual.append(datos_oracion)

    # Guardado por bloques
    if len(bloque_actual) >= TAMANO_BLOQUE:
        archivo_salida = os.path.join(output_dir, f"semeval_bloque_{num_bloque}.npy")
        np.save(archivo_salida, np.array(bloque_actual, dtype=object))
        print(f"Bloque {num_bloque} guardado con {len(bloque_actual)} relaciones.")
        bloque_actual = []
        num_bloque += 1

# Guardar lo que sobre al final
if len(bloque_actual) > 0:
    archivo_salida = os.path.join(output_dir, f"semeval_bloque_{num_bloque}.npy")
    np.save(archivo_salida, np.array(bloque_actual, dtype=object))
    print(f"Bloque final {num_bloque} guardado con {len(bloque_actual)} relaciones.")

print("Extracción completada.")