import string
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader
from torch_cka import CKA
import matplotlib.pyplot as plt
import seaborn as sns

# Lectura del TXT
def cargar_cuestionario_txt(ruta_archivo):
    print(f"Leyendo dataset desde: {ruta_archivo}")
    dataset = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        bloques = f.read().strip().split('\n\n')
        
    for bloque in bloques:
        lineas = [l.strip() for l in bloque.split('\n') if l.strip()]
        if len(lineas) < 2: continue
        oracion_orig = lineas[0]
        for oracion_mask_raw in lineas[1:]:
            oracion_mask = oracion_mask_raw.replace('[MASK]', '[MASK]')
            dataset.append(oracion_mask) # Guardamos solo la oración enmascarada
            
    print(f"Cargadas {len(dataset)} oraciones.")
    return dataset

# Wrapper del modelo para CKA
class ModelForCKA(nn.Module):
    def __init__(self, model_name, mask_token_id):
        super().__init__()
        # Cargamos el cuerpo del modelo
        self.hf_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        # Creamos 13 capas vacías para que torch-cka las "intercepte"
        self.cka_layers = nn.ModuleList([nn.Identity() for _ in range(13)])
        # Guardamos el ID del token que queremos buscar (MASK)
        self.mask_token_id = mask_token_id

    def forward(self, input_ids):
        # Generamos la máscara de atención (con pad_token = 0)
        attention_mask = (input_ids != 0).long()
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Buscamos dinámicamente la posición del token [MASK] en cada frase del batch
        mask_positions = (input_ids == self.mask_token_id).int().argmax(dim=1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        
        # Iteramos por las 13 capas de salida
        for i, state in enumerate(outputs.hidden_states):
            # Extraemos el token [MASK] de forma dinámica
            mask_token_embedding = state[batch_indices, mask_positions, :] 
            # Lo pasamos por nuestra capa chivato para que torch-cka lo capture
            self.cka_layers[i](mask_token_embedding)
            
        return None # No necesitamos devolver nada, torch-cka captura por los "hooks"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 

ruta_txt = "gamb.txt"
oraciones = cargar_cuestionario_txt(ruta_txt)

# Tokenizamos usando el tokenizador base (comparten vocabulario)
print("Tokenizando oraciones.")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(oraciones, padding=True, truncation=True, return_tensors="pt")
mask_id = tokenizer.mask_token_id

# torch-cka requiere un DataLoader estándar de PyTorch que devuelva tuplas (input, etiqueta_ficticia)
dataset = TensorDataset(inputs.input_ids, torch.zeros(len(oraciones))) 
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Inicializamos nuestros wrappers
print("\nCargando modelos.")
modelo_base = ModelForCKA("bert-base-uncased", mask_id).to(device).eval()
modelo_dominio = ModelForCKA("citiusLTL/GambBERT", mask_id).to(device).eval()

# Definimos los nombres de las capas a las que torch-cka debe engancharse
layer_names = [f"cka_layers.{i}" for i in range(13)]

# CKA INTRAMODELO
print("\nCalculando CKA Intramodelo (GambBERT vs GambBERT)")
cka_intra_obj = CKA(
    modelo_dominio, modelo_dominio,
    model1_name="GambBERT", model2_name="GambBERT",
    model1_layers=layer_names, model2_layers=layer_names,
    device=device
)
cka_intra_obj.compare(dataloader, dataloader)
# Extraemos la matriz de resultados (tensor de PyTorch a array de NumPy)
matriz_intra = cka_intra_obj.export()['CKA'].cpu().numpy()

# CKA INTERMODELO
print("Calculando CKA Intermodelo (BERT Base vs GambBERT).")
cka_inter_obj = CKA(
    modelo_base, modelo_dominio,
    model1_name="BERT_Base", model2_name="GambBERT",
    model1_layers=layer_names, model2_layers=layer_names,
    device=device
)
cka_inter_obj.compare(dataloader, dataloader)
# Extraemos la matriz
matriz_inter = cka_inter_obj.export()['CKA'].cpu().numpy()

print("\nGenerando imagen combinada")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Títulos limpios sin las rutas largas
nombre_dominio = "GambBERT"

# Dibujo Intramodelo
sns.heatmap(matriz_intra, ax=axes[0], cmap='gist_earth_r', vmin=0, vmax=1, 
            cbar_kws={'label': 'CKA Similarity'})
axes[0].set_title(f'Intramodelo: {nombre_dominio}', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Capa')
axes[0].set_ylabel('Capa')
axes[0].invert_yaxis()

# Dibujo Intermodelo
sns.heatmap(matriz_inter, ax=axes[1], cmap='gist_earth_r', vmin=0, vmax=1, 
            cbar_kws={'label': 'CKA Similarity'})
axes[1].set_title(f'Intermodelo: BERT vs {nombre_dominio}', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'Capas ({nombre_dominio})')
axes[1].set_ylabel('Capas (BERT Base)')
axes[1].invert_yaxis()

plt.tight_layout()

# Guardamos la imagen final combinada en alta resolución
plt.savefig("cka_resultados_combinados_gamb_mask.png", dpi=300, bbox_inches='tight')

# La mostramos en pantalla
plt.show()