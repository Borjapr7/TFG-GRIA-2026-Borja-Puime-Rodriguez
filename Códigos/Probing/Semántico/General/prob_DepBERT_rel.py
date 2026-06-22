import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_capas = 13
num_bloques = 16
directorio = ""

X_por_capa = {i: [] for i in range(num_capas)}
y_labels = []

print("Cargando y unificando datos de SemEval.")

for b in range(num_bloques):
    ruta_archivo = os.path.join(directorio, f"semeval_bloque_{b}.npy")
    
    if not os.path.exists(ruta_archivo):
        print(f"Aviso: No se encuentra el archivo {ruta_archivo}")
        continue

    try:
        bloque_oraciones = np.load(ruta_archivo, allow_pickle=True)
    except EOFError:
        print(f"El archivo 'semeval_bloque_{b}.npy' está CORRUPTO o INCOMPLETO.")
        continue 
    
    for oracion in bloque_oraciones:
        y_labels.append(oracion['relation_label'])
        
        diccionario_embs = oracion['embeddings']
        for i in range(num_capas):
            emb_cls_capa = diccionario_embs[f'capa_{i}']
            X_por_capa[i].append(emb_cls_capa)
            
print("Lectura de bloques finalizada.")

print("Preparando matrices para entrenamiento.")
y_total = np.array(y_labels)

for i in range(num_capas):
    X_por_capa[i] = np.vstack(X_por_capa[i])

print(f"Total de oraciones para entrenar: {len(y_total)}")

clases, conteos = np.unique(y_total, return_counts=True)

print("DISTRIBUCIÓN DE CLASES")
for c, n in zip(clases, conteos):
    if n < 5:
        print(f"CLASE RARA DETECTADA -> '{c}': {n} ejemplos")
    else:
        print(f"{c}: {n} ejemplos")

clases_raras = clases[conteos < 2]

if len(clases_raras) > 0:
    print(f"\n[Acción] Eliminando las clases basura: {clases_raras}")
    
    mascara_validos = ~np.isin(y_total, clases_raras)
    
    y_total = y_total[mascara_validos]
    
    for i in range(num_capas):
        X_por_capa[i] = X_por_capa[i][mascara_validos]
        
    print(f"Total de oraciones tras la limpieza: {len(y_total)}")
else:
    print("No se detectaron clases problemáticas. El dataset está limpio.")

resultados_accuracy = []

print("Iniciando Regresión Logística por capa.")

for capa in range(num_capas):
    X_capa = X_por_capa[capa]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_capa, y_total, test_size=0.2, random_state=42, stratify=y_total
    )
    
    clf = LogisticRegression(max_iter=3000, n_jobs=-1, solver='lbfgs', verbose=0)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados_accuracy.append(acc)
    
    print(f"Capa {capa:02d} | Accuracy: {acc:.4f}")

