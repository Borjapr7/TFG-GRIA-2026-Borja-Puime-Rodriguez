import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

directorio = "" 

print("Cargando etiquetas de WiC.")
ruta_labels = os.path.join(directorio, "wic_labels.npy")
y_total = np.load(ruta_labels)
print(f"Total de ejemplos: {len(y_total)}")

num_capas = 13
resultados_accuracy = []

print("Iniciando entrenamiento de la Regresión Logística (WSD).")

for capa in range(num_capas):
    ruta_capa = os.path.join(directorio, f"wic_features_capa_{capa}.npy")
    X_capa = np.load(ruta_capa)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_capa, y_total, test_size=0.2, random_state=42, stratify=y_total
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(C=0.001, max_iter=1000, n_jobs=-1, solver='lbfgs', verbose=0)
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    resultados_accuracy.append(acc)
    
    print(f"Capa {capa:02d} | Accuracy: {acc:.4f}")

