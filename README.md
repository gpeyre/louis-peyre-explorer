# Louis Peyre Explorer

Pipeline de vision par ordinateur haute performance pour analyser automatiquement des tableaux numérisés, détecter des sujets en **open vocabulary**, segmenter précisément les régions, produire des crops normalisés `128x128`, et générer un fichier de provenance.

## 1. Installation

Prérequis:
- Python 3.10+
- (Recommandé) GPU CUDA pour accélérer Grounding DINO, SAM ViT-H et CLIP

Installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Important:
- Utiliser un environnement virtuel dédié au projet.
- Eviter de mélanger paquets `conda` et `pip` dans le même environnement, car cela crée souvent des conflits binaires (`numpy/scipy/scikit-learn`).

## 2. Téléchargement automatique des poids

Aucune action manuelle n'est nécessaire.

Au premier lancement, `pipeline.py` télécharge automatiquement les poids Hugging Face suivants:
- `IDEA-Research/grounding-dino-base`
- `facebook/sam-vit-huge`
- `openai/clip-vit-large-patch14`

Les poids sont mis en cache local, puis réutilisés aux exécutions suivantes.

## 3. Lancement

Le pipeline cherche par défaut les images dans:
1. `./images`
2. sinon `./imgs`

Exemple standard:

```bash
python pipeline.py
```

Exemple avec chemins explicites:

```bash
python pipeline.py --input_dir ./images --output_dir ./crops --output_csv ./provenance.csv
```

Limiter le nombre d'images (ex: 20 premières) :

```bash
python pipeline.py --max_images 20
```

Mode Colab + Google Drive (dossier `louis-peyre-images`) :

```bash
python pipeline.py --drive_folder louis-peyre-images --max_images 20
```

## 4. Entrées / sorties

Entrée:
- `./images/*.jpg` (ou `./imgs/*.jpg`)
- Extensions supportées: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Sorties:
- Dossier `./crops/` contenant des fichiers:
  - `crops/{image_id}_{crop_id}.png`
- Fichier `./provenance.csv`

## 5. Format du CSV

Colonnes générées:
- `image_id`
- `crop_id`
- `label`
- `box_x1`
- `box_y1`
- `box_x2`
- `box_y2`
- `score_clip`

## 6. Description du pipeline

1. Détection open vocabulary avec **Grounding DINO**
- Caption large utilisée:
  - `all visible objects, people, animals, body parts, faces, architectural elements, furniture, plants, tools, instruments, decorative elements, symbols`
- Aucune petite liste fermée de labels n'est utilisée pour la détection.

2. Raffinement spatial avec **SAM ViT-H**
- Chaque détection est convertie en masque précis.
- La bounding box est recalculée à partir du masque.

3. Étiquetage sémantique avec **CLIP**
- Vocabulaire étendu + enrichissement dynamique via les sorties DINO.
- Labels multiples autorisés si les scores sont proches.
- Rejet uniquement si `score_clip < 0.20`.

4. Génération des crops
- Bounding box masquée -> carré centré.
- Crop extrait, converti RGB, redimensionné en `128x128`, sauvegardé en PNG.

5. Provenance
- Une ligne CSV par crop retenu avec les coordonnées et le score CLIP.

## 7. Options principales

```bash
python pipeline.py \
  --input_dir ./images \
  --output_dir ./crops \
  --output_csv ./provenance.csv \
  --drive_folder louis-peyre-images \
  --box_threshold 0.22 \
  --text_threshold 0.20 \
  --nms_iou 0.45 \
  --clip_threshold 0.20 \
  --crop_size 128 \
  --crop_padding 0.10 \
  --face_crop_padding 0.28 \
  --clip_batch_size 8 \
  --grounding_model_id IDEA-Research/grounding-dino-base \
  --sam_model_id facebook/sam-vit-huge \
  --clip_model_id openai/clip-vit-large-patch14 \
  --max_images 50 \
  --seed 42
```

## 8. Robustesse et reproductibilité

- Détection automatique CPU/GPU.
- Seed déterministe (`random`, `numpy`, `torch`, `cudnn`).
- Barre de progression via `tqdm`.
- `logging` structuré avec sorties console claires sur chaque étape.
- `try/except` par image pour éviter l'arrêt global en cas d'erreur isolée.

## 9. Dépannage (erreur numpy/scipy/sklearn)

Si vous voyez une erreur du type:
- `AttributeError: _ARRAY_API not found`
- `ImportError: numpy.core.multiarray failed to import`

Alors votre environnement a un conflit de versions binaires. Exécutez:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

Puis relancez:

```bash
python pipeline.py --max_images 5
```

Cas Google Colab:
- Si vous voyez des conflits `numpy` pendant `pip install -r requirements.txt`, c'est normal: ce fichier est orienté environnement local.
- Sur Colab, utilisez `requirements-colab.txt` (prévu pour `numpy>=2`):

```bash
!pip install -q -r requirements-colab.txt
```

- Si vous voyez `operator torchvision::nms does not exist`, forcez un couple cohérent `torch/torchvision/torchaudio`:

```bash
!pip uninstall -y torch torchvision torchaudio transformers accelerate || true
!pip install -q --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
!pip install -q --no-cache-dir -r requirements-colab.txt
```

## 10. Exécution sur Google Colab

1. Ouvrir un notebook Colab.
2. Activer un GPU: `Runtime > Change runtime type > T4 GPU` (ou équivalent).
3. Exécuter les cellules suivantes:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!rm -rf /content/louis-peyre-explorer
!git clone https://github.com/gpeyre/louis-peyre-explorer.git
%cd /content/louis-peyre-explorer
!pip install -q --upgrade pip
!pip uninstall -y torch torchvision torchaudio transformers accelerate || true
!pip install -q --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
!pip install -q --no-cache-dir -r requirements-colab.txt
```

```bash
# Le dossier d'images doit être:
# /content/drive/MyDrive/louis-peyre-images
!python pipeline.py --drive_folder louis-peyre-images --max_images 20
```

4. Résultats générés sur Drive:
- `/content/drive/MyDrive/louis-peyre-images/crops/`
- `/content/drive/MyDrive/louis-peyre-images/provenance.csv`

Note mémoire Colab:
- `sam-vit-huge` peut être trop lourd selon la machine GPU.
- En cas d'OOM, utiliser un modèle SAM plus léger:

```bash
!python pipeline.py --drive_folder louis-peyre-images --sam_model_id facebook/sam-vit-base --max_images 20
```

Messages fréquents en Colab:
- `CLIPModel LOAD REPORT ... UNEXPECTED ... position_ids`
  - Ce message est généralement non bloquant dans ce pipeline. Le chargement CLIP peut continuer normalement.
- `GroundingDinoProcessor.post_process_grounded_object_detection() got an unexpected keyword argument 'box_threshold'`
  - Ce problème venait d'une différence d'API `transformers` entre versions.
  - Le script est maintenant compatible avec `box_threshold` et `threshold`.
  - Si vous voyez encore cette erreur, assurez-vous d'avoir la dernière version du repo dans Colab:

```bash
%cd /content
!rm -rf /content/louis-peyre-explorer
!git clone https://github.com/gpeyre/louis-peyre-explorer.git
%cd /content/louis-peyre-explorer
!pip install -q -r requirements-colab.txt
```
