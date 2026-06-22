# Desenmascarando Modelos Grandes de Linguaxe: "Probing" de Modelos Adaptados ao Domínio da Saúde Mental

Este repositorio contiene el código fuente, los scripts de experimentación y los conjuntos de datos empleados para el Trabajo de Fin de Grado (TFG) en Inteligencia Artificial por la **Universidade de Santiago de Compostela (USC)**.

El objetivo principal de este proyecto es auditar la "caja negra" de arquitecturas Transformer (familia BERT) que han sido adaptadas para la detección de trastornos de salud mental (depresión, anorexia, ludopatía y autolesiones). Mediante técnicas de *probing* lingüístico y análisis geométrico (CKA), se investiga cómo la inyección de vocabulario clínico altera las representaciones latentes y si produce un olvido de las capacidades gramaticales base.

---

## Estructura del Repositorio

El proyecto está organizado en dos grandes bloques funcionales: `Datos` (conjuntos de evaluación) y `Códigos` (scripts de experimentación y análisis).

```text
Repositorio/
├── Datos/
│   ├── Cuestionarios/   # Plantillas generativas (MLM) de BDI, EDQ, DSM-V y SH
│   ├── conll2003/       # Corpus para probing sintáctico (POS y Chunking) y NER
│   ├── semEval/         # Corpus (SemEval-2010 Task 8) para extracción de relaciones
│   └── WiC/             # Corpus (Word-in-Context) para desambiguación semántica
│
└── Códigos/
    ├── CKA/             # Análisis de similitud topológica (Alineación de Kernels Centrados)
    ├── Probing/         # Extracción de representaciones latentes y clasificadores
    │   ├── Sintáctico/  # Scripts de regresión logística para POS y Chunking
    │   └── Semántico/   
    │       ├── General/ # Tareas de NER, Relaciones y WSD sobre representaciones congeladas
    │       └── Dominio/ # Scripts generativos (MRR@Top-5) sobre cuestionarios clínicos
    └── GLUE/            # Evaluación de la línea base de rendimiento de propósito general
