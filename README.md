# Potato Dry Matter Optics ML

Sistema de visió artificial i machine learning per estimar la matèria seca (%MS) i detectar defectes en patates, utilitzant informació òptica en el rang visible (RGB) i banda NIR (970 nm).

L'objectiu principal no és obtenir una predicció perfecta de la matèria seca, sinó construir un model de predicció que permeti estudiar si existeix una relació entre dades òptiques de baix cost d'obtenció i la matèria seca, amb la finalitat de desenvolupar tecnologia econòmica orientada a pimes del sector agroalimentari.

El projecte també planteja la construcció d'un pipeline complet com a prototip funcional d'automatització del control de qualitat.

Projecte desenvolupat per a l'assignatura Projectes d'Enginyeria.

---

## Objectius del projecte

### 1) Model per estudiar la relació RGB/NIR - %MS

- Preprocessar imatges per extreure característiques rellevants RGB.
- Estudiar la relació entre color (RGB) + informació puntual NIR (970 nm) i %MS.
- Entrenar un model de regressió (MLP).
- Validar amb MAPE i R2 com a mètriques principals.

### 2) Detecció de defectes (Proof of Concept)

- Detecció mitjançant crida a una API de Roboflow.
- Model desplegat externament (consumit via API).
- Integració en pipeline automatitzat.
- Detecció de:
  - Color verdós.
  - Esquerdes.
  - Taques.
  - Floridura.
  - Brots.

### 3) Pipeline complet (prototip)

El projecte integra:

- Captura d'imatges RGB i informació NIR.
- Preprocessament.
- Detecció de defectes.
- Predicció de %MS.
- Generació d'un output estructurat i senzill.

Amb l'objectiu de construir un pipeline complet d'automatització com a prototip funcional.

---

## Instal·lació

### 1) Crear i activar l'entorn de Conda

```bash
conda env create -f envs/environment.yml
conda activate quality_pipeline_env
```

Instal·lació extra (amb l'entorn activat):

```bash
pip install --force-reinstall --no-cache-dir numpy==1.24.3
```

### 2) Configurar la clau de Roboflow

Crea un fitxer `.env` a l'arrel del projecte amb:

```env
ROBOFLOW_API_KEY=la_teva_api_key
```