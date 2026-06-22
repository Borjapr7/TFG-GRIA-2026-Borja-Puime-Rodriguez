import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_capas = 13
num_bloques = 6 
directorio = ""

X_por_capa = {i: [] for i in range(num_capas)}
y_labels = []

print("Cargando y unificando bloques de datos.")

for b in range(num_bloques):
    ruta_archivo = os.path.join(directorio, f"conll_bloque_{b}.npy")
    
    if not os.path.exists(ruta_archivo):
        print(f" Salto bloque {b}: archivo no encontrado.")
        continue

    bloque_oraciones = np.load(ruta_archivo, allow_pickle=True)
    
    print(f"Procesando bloque {b}.")
    for oracion in bloque_oraciones:
        etiquetas_oracion = oracion['pos_labels']
        y_labels.extend(etiquetas_oracion)
        
        diccionario_embs = oracion['embeddings']
        
        for i in range(num_capas):
            emb_capa_oracion = diccionario_embs[f'capa_{i}']
            X_por_capa[i].append(emb_capa_oracion)

print("Concatenando datos finales.")
y_total = np.array(y_labels)

for i in range(num_capas):
    X_por_capa[i] = np.vstack(X_por_capa[i])

print(f"Datos cargados. Total de tokens: {len(y_total)}")
resultados_accuracy = []

for capa in range(num_capas):
    X_capa = X_por_capa[capa]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_capa, y_total, test_size=0.2, random_state=42
    )
    
    clf = LogisticRegression(max_iter=500, n_jobs=-1, solver='lbfgs', verbose=0)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    resultados_accuracy.append(acc)
    print(f"Capa {capa:02d} | Accuracy: {acc:.4f}")

