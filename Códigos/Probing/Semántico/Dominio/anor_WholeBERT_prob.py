import torch
import string
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Función de lectura del TXT
def cargar_cuestionario_txt(ruta_archivo):
    print(f"Leyendo dataset desde: {ruta_archivo}")
    dataset = []
    
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        # Separamos por bloques vacíos
        bloques = f.read().strip().split('\n\n')
        
    for bloque in bloques:
        # Limpiamos líneas vacías que puedan colarse dentro del bloque
        lineas = [l.strip() for l in bloque.split('\n') if l.strip()]
        
        # Si el bloque tiene menos de 2 líneas, no hay nada que comparar
        if len(lineas) < 2:
            continue
            
        oracion_orig = lineas[0]
        
        # Iteramos sobre todas las versiones enmascaradas de este bloque
        for oracion_mask_raw in lineas[1:]:
            oracion_mask = oracion_mask_raw.replace('[MASK]', '[MASK]')
            
            tokens_orig = oracion_orig.split()
            tokens_mask = oracion_mask.split()
            target_word = None
            
            for o_w, m_w in zip(tokens_orig, tokens_mask):
                if "[MASK]" in m_w:
                    target_word = o_w.translate(str.maketrans('', '', string.punctuation))
                    break
            
            if target_word:
                dataset.append({
                    "original": oracion_orig,
                    "masked": oracion_mask,
                    "target": target_word
                })
                
    print(f"Cargados {len(dataset)} items.")
    return dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name = "citiusLTL/WholeBERT" 
print(f"Cargando modelo: {model_name}.")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
model.to(device)
model.eval()

ruta_txt = "" 
dataset = cargar_cuestionario_txt(ruta_txt)

print(dataset[0])

# Diccionario para guardar el Reciprocal Rank (RR) de todas las oraciones, por capa
resultados_rr_por_capa = {i: [] for i in range(13)}

print("\nIniciando extracción")

for item in dataset:
    # Obtenemos el ID de la palabra correcta en el vocabulario del modelo
    target_token_ids = tokenizer.encode(item["target"], add_special_tokens=False)
    target_id = target_token_ids[0] 
    
    # Pasamos la oración con [MASK] por el modelo
    inputs = tokenizer(item["masked"], return_tensors="pt").to(device)
    
    # Buscamos en qué posición exacta cayó el token [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.hidden_states # Tupla con 13 tensores
    
    # Evaluamos capa por capa usando la cabeza de MLM
    for capa_idx, estado_capa in enumerate(hidden_states):
        # Introducimos el estado de esta capa específica a la cabeza predictora
        logits_capa = model.cls(estado_capa)
        
        # Extraemos las predicciones (Top 5) para la posición del [MASK]
        mask_logits = logits_capa[0, mask_token_index, :]
        top_5_ids = torch.topk(mask_logits, 5, dim=-1).indices.squeeze().tolist()
        
        if type(top_5_ids) is not list:
            top_5_ids = [top_5_ids]
            
        # Calculamos el Reciprocal Rank (RR) para esta capa
        if target_id in top_5_ids:
            rank = top_5_ids.index(target_id) + 1 # +1 porque los índices empiezan en 0
            rr = 1.0 / rank
        else:
            rr = 0.0 # Si no está en el Top 5, la puntuación es 0
            
        resultados_rr_por_capa[capa_idx].append(rr)

# CÁLCULO DEL MRR (Mean Reciprocal Rank) FINAL
print("\n RESULTADOS FINALES (MRR @ Top-5)")
mrr_por_capa = []

for capa in range(13):
    # La media de todos los RRs de esta capa
    mrr = sum(resultados_rr_por_capa[capa]) / len(resultados_rr_por_capa[capa])
    mrr_por_capa.append(mrr)
    print(f"Capa {capa:02d} | MRR: {mrr:.4f}")

# ANÁLISIS VISUAL DE PREDICCIONES (TOP 5)
print("\nANÁLISIS VISUAL DE PREDICCIONES (TOP 5)")

mejor_capa = mrr_por_capa.index(max(mrr_por_capa))
print(f"Mejor capa detectada: Capa {mejor_capa} (MRR: {mrr_por_capa[mejor_capa]:.4f})\n")

num_ejemplos_a_mostrar = min(20, len(dataset))

for i in range(num_ejemplos_a_mostrar):
    item = dataset[i]
    print(f"Oración masked:   {item['masked']}")
    print(f"Palabra objetivo: '{item['target']}'")
    
    inputs = tokenizer(item["masked"], return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    estado_mejor_capa = outputs.hidden_states[mejor_capa]
    logits = model.cls(estado_mejor_capa)
    mask_logits = logits[0, mask_token_index, :]
    
    top_5_ids = torch.topk(mask_logits, 5, dim=-1).indices.squeeze().tolist()
    if type(top_5_ids) is not list: top_5_ids = [top_5_ids]
        
    top_5_palabras = [tokenizer.decode([token_id]).strip() for token_id in top_5_ids]
    
    print(f"Top 5 del modelo: {top_5_palabras}")
    print("-" * 50)