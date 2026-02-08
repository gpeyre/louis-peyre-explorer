#!/usr/bin/env python3
"""Pipeline SOTA de vision par ordinateur pour analyser des tableaux numérisés.

Étapes:
1. Détection open-vocabulary avec Grounding DINO.
2. Raffinement spatial avec SAM (masques précis).
3. Attribution sémantique avec CLIP (vocabulaire étendu et dynamique).
4. Export des crops carrés 128x128 + provenance CSV.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        CLIPModel,
        CLIPProcessor,
        SamModel,
        SamProcessor,
    )
except Exception as exc:  # noqa: BLE001
    message = (
        "Echec d'import de transformers (souvent conflit numpy/scipy/scikit-learn).\\n"
        "Cause detectee: "
        f"{type(exc).__name__}: {exc}\\n\\n"
        "Correction recommandee (dans un environnement virtuel propre):\\n"
        "  pip install --upgrade pip\\n"
        "  pip install --force-reinstall -r requirements.txt\\n\\n"
        "Si vous utilisez conda, evitez de melanger conda et pip dans la meme base.\\n"
        "Creez un venv dedie au projet puis relancez."
    )
    print(message, file=sys.stderr)
    raise SystemExit(1) from exc


CAPTION_OPEN_VOCAB = (
    "all visible objects, people, animals, body parts, faces, "
    "architectural elements, furniture, plants, tools, instruments, "
    "decorative elements, symbols"
)

VOCABULAIRE_BASE = [
    "personne",
    "visage",
    "main",
    "bras",
    "jambe",
    "pied",
    "animal",
    "oiseau",
    "cheval",
    "chien",
    "chat",
    "poisson",
    "arbre",
    "plante",
    "fleur",
    "feuille",
    "fenetre",
    "porte",
    "mur",
    "colonne",
    "toit",
    "table",
    "chaise",
    "lit",
    "livre",
    "instrument",
    "violon",
    "guitare",
    "objet",
    "vase",
    "drape",
    "vetement",
    "bijou",
    "ciel",
    "nuage",
    "eau",
    "rivage",
    "batiment",
    "maison",
    "tour",
    "route",
    "pont",
    "barque",
    "lampe",
    "miroir",
    "rideau",
    "ornement",
    "symbole",
    "couronne",
    "arme",
    "bouclier",
    "poterie",
    "fruit",
    "bol",
    "verre",
    "montagne",
    "colline",
    "rocher",
    "sol",
    "ombre",
    "lumiere",
]


@dataclass
class DetectionRegion:
    boite: list[float]  # [x1, y1, x2, y2]
    score_detection: float
    etiquette_dino: str


@dataclass
class RegionFinale:
    masque_boite: list[int]
    etiquette: str
    score_clip: float
    crop: Image.Image


class PipelinePeyre:
    def __init__(
        self,
        appareil: torch.device,
        seuil_boite: float,
        seuil_texte: float,
        seuil_clip_min: float,
        taille_crop: int,
        lot_clip: int,
        modele_dino_id: str,
        modele_sam_id: str,
        modele_clip_id: str,
    ) -> None:
        self.appareil = appareil
        self.seuil_boite = seuil_boite
        self.seuil_texte = seuil_texte
        self.seuil_clip_min = seuil_clip_min
        self.taille_crop = taille_crop
        self.lot_clip = lot_clip

        logging.info("Chargement Grounding DINO: %s", modele_dino_id)
        self.proc_dino = AutoProcessor.from_pretrained(modele_dino_id)
        self.modele_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            modele_dino_id
        ).to(self.appareil)
        self.modele_dino.eval()

        logging.info("Chargement SAM: %s", modele_sam_id)
        self.proc_sam = SamProcessor.from_pretrained(modele_sam_id)
        self.modele_sam = SamModel.from_pretrained(modele_sam_id).to(self.appareil)
        self.modele_sam.eval()

        logging.info("Chargement CLIP: %s", modele_clip_id)
        self.proc_clip = CLIPProcessor.from_pretrained(modele_clip_id)
        self.modele_clip = CLIPModel.from_pretrained(modele_clip_id).to(self.appareil)
        self.modele_clip.eval()

    def detecter_regions(self, image: Image.Image) -> list[DetectionRegion]:
        entrees = self.proc_dino(images=image, text=CAPTION_OPEN_VOCAB, return_tensors="pt").to(
            self.appareil
        )
        with torch.no_grad():
            sorties = self.modele_dino(**entrees)

        # Compatibilité multi-versions transformers:
        # - anciennes versions: box_threshold
        # - versions récentes: threshold
        try:
            resultats = self.proc_dino.post_process_grounded_object_detection(
                sorties,
                entrees.input_ids,
                box_threshold=self.seuil_boite,
                text_threshold=self.seuil_texte,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            resultats = self.proc_dino.post_process_grounded_object_detection(
                sorties,
                entrees.input_ids,
                threshold=self.seuil_boite,
                text_threshold=self.seuil_texte,
                target_sizes=[image.size[::-1]],
            )[0]

        boites = resultats["boxes"].detach().cpu().tolist()
        scores = resultats["scores"].detach().cpu().tolist()
        etiquettes_brutes = resultats.get("labels")
        if etiquettes_brutes is None:
            etiquettes_brutes = resultats.get("text_labels", [])
        etiquettes = [str(e) for e in etiquettes_brutes]

        regions: list[DetectionRegion] = []
        for boite, score, etiquette in zip(boites, scores, etiquettes):
            regions.append(
                DetectionRegion(
                    boite=[float(v) for v in boite],
                    score_detection=float(score),
                    etiquette_dino=etiquette.strip(),
                )
            )
        return regions

    def segmenter_boites(self, image: Image.Image, regions: Sequence[DetectionRegion]) -> list[list[int]]:
        if not regions:
            return []

        boites = [[r.boite for r in regions]]
        entrees = self.proc_sam(
            image,
            input_boxes=boites,
            return_tensors="pt",
        ).to(self.appareil)

        with torch.no_grad():
            sorties = self.modele_sam(**entrees, multimask_output=True)

        masques = self.proc_sam.image_processor.post_process_masks(
            sorties.pred_masks.detach().cpu(),
            entrees["original_sizes"].detach().cpu(),
            entrees["reshaped_input_sizes"].detach().cpu(),
        )

        iou_scores = sorties.iou_scores.detach().cpu()
        masques_image = masques[0]
        boites_masquees: list[list[int]] = []

        for i in range(len(regions)):
            idx_meilleur = int(torch.argmax(iou_scores[0, i]).item())
            masque = masques_image[i, idx_meilleur].numpy() > 0
            ys, xs = np.where(masque)

            if len(xs) == 0 or len(ys) == 0:
                x1, y1, x2, y2 = [int(round(v)) for v in regions[i].boite]
            else:
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

            boites_masquees.append([x1, y1, x2, y2])

        return boites_masquees

    def etiqueter_regions(
        self,
        image: Image.Image,
        boites_masquees: Sequence[list[int]],
        etiquettes_dino: Sequence[str],
    ) -> list[RegionFinale]:
        if not boites_masquees:
            return []

        vocabulaire = self.generer_vocabulaire_dynamique(etiquettes_dino)
        textes = [f"tableau représentant {mot}" for mot in vocabulaire]

        regions_finales: list[RegionFinale] = []
        crops_originaux: list[Image.Image] = []
        boites_carrees: list[list[int]] = []

        largeur, hauteur = image.size
        for boite in boites_masquees:
            carre = convertir_boite_en_carre(boite, largeur, hauteur)
            boites_carrees.append(carre)
            x1, y1, x2, y2 = carre
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crops_originaux.append(crop)

        for i in range(0, len(crops_originaux), self.lot_clip):
            lot_crops = crops_originaux[i : i + self.lot_clip]
            entrees = self.proc_clip(
                text=textes,
                images=lot_crops,
                return_tensors="pt",
                padding=True,
            ).to(self.appareil)

            with torch.no_grad():
                sorties = self.modele_clip(**entrees)

            logits = sorties.logits_per_image
            probas = logits.softmax(dim=-1).detach().cpu().numpy()

            for j, vecteur in enumerate(probas):
                idx_global = i + j
                meilleur_idx = int(np.argmax(vecteur))
                meilleur_score = float(vecteur[meilleur_idx])
                if meilleur_score < self.seuil_clip_min:
                    continue

                etiquette = construire_etiquette_multi(vocabulaire, vecteur)
                crop_128 = lot_crops[j].resize(
                    (self.taille_crop, self.taille_crop),
                    resample=Image.Resampling.LANCZOS,
                )
                regions_finales.append(
                    RegionFinale(
                        masque_boite=boites_carrees[idx_global],
                        etiquette=etiquette,
                        score_clip=meilleur_score,
                        crop=crop_128,
                    )
                )

        return regions_finales

    @staticmethod
    def generer_vocabulaire_dynamique(etiquettes_dino: Sequence[str]) -> list[str]:
        mots = set(VOCABULAIRE_BASE)

        for etiquette in etiquettes_dino:
            t = normaliser_texte(etiquette)
            if not t:
                continue
            mots.add(t)
            for fragment in t.replace("_", " ").split():
                if len(fragment) >= 3:
                    mots.add(fragment)

        # Ajout de classes génériques utiles dans les tableaux figuratifs.
        mots.update(["silhouette", "portrait", "scene", "ornemental", "nature", "abstrait"])

        return sorted(mots)


def normaliser_texte(texte: str) -> str:
    t = texte.strip().lower()
    remplacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "à": "a",
        "ù": "u",
        "î": "i",
        "ô": "o",
        "ç": "c",
        "'": " ",
        '"': " ",
    }
    for src, dst in remplacements.items():
        t = t.replace(src, dst)
    t = " ".join(t.split())
    return t


def construire_etiquette_multi(vocabulaire: Sequence[str], scores: np.ndarray) -> str:
    idx_tries = np.argsort(scores)[::-1]
    meilleur = float(scores[idx_tries[0]])
    seuil_proche = max(0.05, 0.85 * meilleur)

    labels: list[str] = []
    for idx in idx_tries[:5]:
        score = float(scores[idx])
        if score < seuil_proche:
            break
        labels.append(vocabulaire[int(idx)])

    return "|".join(labels) if labels else vocabulaire[int(idx_tries[0])]


def convertir_boite_en_carre(boite: Sequence[int | float], largeur: int, hauteur: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in boite]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cote = max(w, h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nx1 = int(math.floor(cx - cote / 2.0))
    ny1 = int(math.floor(cy - cote / 2.0))
    nx2 = int(math.ceil(cx + cote / 2.0))
    ny2 = int(math.ceil(cy + cote / 2.0))

    if nx1 < 0:
        nx2 += -nx1
        nx1 = 0
    if ny1 < 0:
        ny2 += -ny1
        ny1 = 0
    if nx2 > largeur:
        decal = nx2 - largeur
        nx1 = max(0, nx1 - decal)
        nx2 = largeur
    if ny2 > hauteur:
        decal = ny2 - hauteur
        ny1 = max(0, ny1 - decal)
        ny2 = hauteur

    nx1 = max(0, min(nx1, largeur - 1))
    ny1 = max(0, min(ny1, hauteur - 1))
    nx2 = max(nx1 + 1, min(nx2, largeur))
    ny2 = max(ny1 + 1, min(ny2, hauteur))

    # Rééquilibrage final pour se rapprocher d'un carré parfait sous contraintes d'image.
    cote_final = min(nx2 - nx1, ny2 - ny1)
    nx2 = nx1 + cote_final
    ny2 = ny1 + cote_final

    return [int(nx1), int(ny1), int(nx2), int(ny2)]


def lister_images(dossier: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in dossier.glob("*") if p.suffix.lower() in extensions])


def est_colab() -> bool:
    return Path("/content").exists()


def choisir_dossier_entree(
    dossier_requis: str | None,
    dossier_drive: str | None,
) -> Path:
    if dossier_drive:
        return Path("/content/drive/MyDrive") / dossier_drive
    if dossier_requis:
        return Path(dossier_requis)

    candidats = []
    if est_colab():
        candidats.extend(
            [
                Path("/content/drive/MyDrive/louis-peyre-images"),
                Path("/content/drive/MyDrive/images"),
                Path("/content/drive/MyDrive/imgs"),
            ]
        )
    candidats.extend([Path("images"), Path("imgs")])
    for dossier in candidats:
        if dossier.exists() and dossier.is_dir():
            return dossier
    return Path("/content/drive/MyDrive/louis-peyre-images") if est_colab() else Path("images")


def resoudre_sorties(
    dossier_entree: Path,
    output_dir_arg: str,
    output_csv_arg: str,
) -> tuple[Path, Path]:
    # En Colab, si les valeurs par défaut sont conservées, on écrit sur Drive.
    if est_colab():
        if output_dir_arg == "crops":
            dossier_sortie = dossier_entree / "crops"
        else:
            dossier_sortie = Path(output_dir_arg)

        if output_csv_arg == "provenance.csv":
            csv_sortie = dossier_entree / "provenance.csv"
        else:
            csv_sortie = Path(output_csv_arg)
        return dossier_sortie, csv_sortie

    return Path(output_dir_arg), Path(output_csv_arg)


def fixer_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configurer_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sauvegarder_provenance(lignes: Iterable[dict], chemin_csv: Path) -> None:
    df = pd.DataFrame(
        lignes,
        columns=[
            "image_id",
            "crop_id",
            "label",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "score_clip",
        ],
    )
    df.to_csv(chemin_csv, index=False)


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Louis Peyre Explorer")
    parser.add_argument("--input_dir", type=str, default=None, help="Dossier des images source")
    parser.add_argument(
        "--drive_folder",
        type=str,
        default=None,
        help=(
            "Dossier relatif sous /content/drive/MyDrive (mode Colab), "
            "ex: louis-peyre-images"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="crops", help="Dossier des crops")
    parser.add_argument(
        "--output_csv", type=str, default="provenance.csv", help="Fichier CSV de provenance"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed déterministe")
    parser.add_argument("--box_threshold", type=float, default=0.22, help="Seuil Grounding DINO")
    parser.add_argument("--text_threshold", type=float, default=0.20, help="Seuil texte DINO")
    parser.add_argument("--clip_threshold", type=float, default=0.20, help="Seuil CLIP")
    parser.add_argument("--crop_size", type=int, default=128, help="Taille de sortie des crops")
    parser.add_argument("--clip_batch_size", type=int, default=8, help="Lot CLIP par image")
    parser.add_argument(
        "--grounding_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Identifiant Hugging Face du modele Grounding DINO",
    )
    parser.add_argument(
        "--sam_model_id",
        type=str,
        default="facebook/sam-vit-huge",
        help="Identifiant Hugging Face du modele SAM",
    )
    parser.add_argument(
        "--clip_model_id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Identifiant Hugging Face du modele CLIP",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Nombre max d'images a traiter (optionnel, utile pour limiter le calcul)",
    )
    return parser.parse_args()


def main() -> None:
    args = parser_args()
    configurer_logging()
    fixer_seed(args.seed)

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Appareil détecté: %s", appareil)
    logging.info("Environnement Colab: %s", "oui" if est_colab() else "non")

    dossier_entree = choisir_dossier_entree(args.input_dir, args.drive_folder)
    dossier_sortie, csv_sortie = resoudre_sorties(dossier_entree, args.output_dir, args.output_csv)

    if not dossier_entree.exists() or not dossier_entree.is_dir():
        raise FileNotFoundError(
            f"Le dossier d'entrée n'existe pas: {dossier_entree} "
            "(attendu: --input_dir, --drive_folder, ./images, ./imgs ou /content/drive/MyDrive/louis-peyre-images)"
        )

    images = lister_images(dossier_entree)
    if not images:
        raise RuntimeError(f"Aucune image trouvée dans {dossier_entree}")
    total_images_disponibles = len(images)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max_images doit etre strictement positif")
        images = images[: args.max_images]

    dossier_sortie.mkdir(parents=True, exist_ok=True)
    logging.info("Dossier d'entree: %s", dossier_entree.resolve())
    logging.info("Dossier de sortie crops: %s", dossier_sortie.resolve())
    logging.info("Fichier CSV de sortie: %s", csv_sortie.resolve())
    logging.info(
        "Images detectees: %d | Images traitees: %d",
        total_images_disponibles,
        len(images),
    )
    logging.info(
        "Seuils -> DINO boite: %.2f | DINO texte: %.2f | CLIP: %.2f",
        args.box_threshold,
        args.text_threshold,
        args.clip_threshold,
    )
    logging.info("Taille crop: %dx%d", args.crop_size, args.crop_size)

    pipeline = PipelinePeyre(
        appareil=appareil,
        seuil_boite=args.box_threshold,
        seuil_texte=args.text_threshold,
        seuil_clip_min=args.clip_threshold,
        taille_crop=args.crop_size,
        lot_clip=args.clip_batch_size,
        modele_dino_id=args.grounding_model_id,
        modele_sam_id=args.sam_model_id,
        modele_clip_id=args.clip_model_id,
    )

    lignes_csv: list[dict] = []

    barre = tqdm(images, desc="Traitement des images", unit="image")
    for chemin_image in barre:
        image_id = chemin_image.stem
        barre.set_postfix_str(image_id)
        try:
            logging.info("Image %s | ouverture", chemin_image.name)
            image = Image.open(chemin_image).convert("RGB")
            logging.info("Image %s | detection Grounding DINO", chemin_image.name)
            regions = pipeline.detecter_regions(image)
            if not regions:
                logging.warning("Aucune détection pour %s", chemin_image.name)
                continue

            logging.info("Image %s | segmentation SAM (%d regions)", chemin_image.name, len(regions))
            boites_masquees = pipeline.segmenter_boites(image, regions)
            etiquettes_dino = [r.etiquette_dino for r in regions]
            logging.info("Image %s | etiquetage CLIP", chemin_image.name)
            regions_finales = pipeline.etiqueter_regions(image, boites_masquees, etiquettes_dino)

            if not regions_finales:
                logging.warning("Aucun crop retenu (seuil CLIP) pour %s", chemin_image.name)
                continue

            logging.info("Image %s | export de %d crops", chemin_image.name, len(regions_finales))
            for idx, region in enumerate(regions_finales, start=1):
                crop_id = f"{idx:04d}"
                nom_crop = f"{image_id}_{crop_id}.png"
                chemin_crop = dossier_sortie / nom_crop
                region.crop.save(chemin_crop, format="PNG")

                x1, y1, x2, y2 = region.masque_boite
                lignes_csv.append(
                    {
                        "image_id": image_id,
                        "crop_id": crop_id,
                        "label": region.etiquette,
                        "box_x1": x1,
                        "box_y1": y1,
                        "box_x2": x2,
                        "box_y2": y2,
                        "score_clip": round(region.score_clip, 6),
                    }
                )
            logging.info("Image %s | terminee", chemin_image.name)

        except Exception as exc:  # noqa: BLE001
            logging.exception("Erreur sur l'image %s: %s", chemin_image.name, exc)

    sauvegarder_provenance(lignes_csv, csv_sortie)
    logging.info("Terminé. %d crops exportés.", len(lignes_csv))
    logging.info("CSV: %s", csv_sortie.resolve())


if __name__ == "__main__":
    main()
