"""
Mòdul: raw_image_treatment
Funcions relacionades amb el preprocés de la il·luminació (RGB i NIR).
"""

from typing import Tuple, Optional, Any, List, Dict
from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient


def potato_defect_classification(image):
    """
    Classifica els defectes de la patata a partir d'una imatge.

    Paràmetres
    ----------
    image : Any
        Objecte imatge ja carregat (per exemple, numpy array, PIL.Image, etc.).
    """
    pass


def potato_pixels_rgb_img(image: Any, margin: int = 0) -> Tuple[Optional[Image.Image], Image.Image]:
    """
    Preprocessa la imatge en el rang visible (RGB) amb el model de Roboflow:
    - Segmenta la patata.
    - Erosiona la màscara amb un marge cap a dins.
    - Genera una imatge de visualització (contorn original + erosionat + bbox).
    - Genera una imatge 'cropped' de la patata (fons negre fora de la màscara).

    Paràmetres
    ----------
    image : str o pathlib.Path
        Ruta a la imatge RGB al disc.
    margin : int, opcional
        Mida de l'erosió (en píxels cap a dins). Per defecte 0.

    Retorna
    -------
    cropped_img : PIL.Image.Image o None
        Imatge retallada de la patata (fora màscara = negre). None si no hi ha màscara.
    vis_img : PIL.Image.Image
        Imatge original amb contorns i bounding box pintats.
    """

    ROBOFLOW_MODEL_ID = "coco-dataset-vdnr1/23"

    # ------------------------------------------------------------------
    # Helpers interns
    # ------------------------------------------------------------------

    def _prediction_to_points(pred: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
        """
        Extreu punts de contorn si existeixen en el JSON de Roboflow.
        """
        pts = (
            pred.get("points")
            or pred.get("polygon")
            or pred.get("vertices")
            or pred.get("segmentation")
        )
        if not pts:
            return None

        if isinstance(pts, list) and len(pts) > 0:
            # Format [{x:..., y:...}, ...]
            if isinstance(pts[0], dict):
                out = []
                for p in pts:
                    x = p.get("x")
                    y = p.get("y")
                    if x is not None and y is not None:
                        out.append((float(x), float(y)))
                return out if len(out) >= 3 else None

            # Format [[x, y], ...] o tuples
            if isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
                out = [(float(p[0]), float(p[1])) for p in pts]
                return out if len(out) >= 3 else None

        return None

    def polygon_to_mask(points: List[Tuple[float, float]], w: int, h: int) -> np.ndarray:
        """
        Rasteritza un polígon a màscara binària.
        """
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        pts_int = [(int(x), int(y)) for x, y in points]
        draw.polygon(pts_int, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _normalize_mask(m: Any, w: int, h: int) -> Optional[np.ndarray]:
        """
        Converteix un camp 'mask' del JSON a np.ndarray (h, w) binari.
        """
        if m is None:
            return None

        if isinstance(m, np.ndarray):
            arr = m
        elif isinstance(m, (list, tuple)):
            arr = np.array(m)
        elif isinstance(m, dict) and "data" in m:
            # per si ve com {data: [...]}
            arr = np.array(m["data"])
        else:
            try:
                arr = np.array(m)
            except Exception:
                return None

        # Assegurar 2D
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] == 1:
                arr = arr[0]

        if arr.ndim != 2:
            return None

        arr = (arr > 0).astype(np.uint8)

        # Ajust de mida si cal
        if arr.shape != (h, w):
            if arr.size == h * w:
                arr = arr.reshape((h, w))
            else:
                mask_img = Image.fromarray(arr * 255)
                mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
                arr = (np.array(mask_img) > 0).astype(np.uint8)

        return arr

    def extract_instance_masks_from_result(result: Dict[str, Any], w: int, h: int) -> List[np.ndarray]:
        """
        Intenta extreure màscares per instància des de result.
        Prioritza:
          1) points -> màscara
          2) mask directe
        """
        masks_local: List[np.ndarray] = []
        preds = result.get("predictions", []) if isinstance(result, dict) else []

        # 1) Si hi ha polígons
        for pred in preds:
            points = _prediction_to_points(pred)
            if points:
                masks_local.append(polygon_to_mask(points, w, h))

        if masks_local:
            return masks_local

        # 2) Si hi ha màscares directes
        for pred in preds:
            nm = _normalize_mask(
                pred.get("mask") or pred.get("segmentation_mask"),
                w,
                h,
            )
            if nm is not None:
                masks_local.append(nm)

        return masks_local

    def mask_boundary(mask: np.ndarray) -> np.ndarray:
        """
        Calcula el contorn d'una màscara binària sense OpenCV.
        """
        m = mask.astype(bool)
        h, w = m.shape

        up = np.zeros_like(m)
        up[1:, :] = m[:-1, :]

        down = np.zeros_like(m)
        down[:-1, :] = m[1:, :]

        left = np.zeros_like(m)
        left[:, 1:] = m[:, :-1]

        right = np.zeros_like(m)
        right[:, :-1] = m[:, 1:]

        interior = up & down & left & right & m
        boundary = m & (~interior)
        return boundary.astype(np.uint8)

    def erode_mask(mask: np.ndarray, margin_int: int) -> np.ndarray:
        """
        Erosiona una màscara binària aplicant un marge cap a dins.
        """
        if margin_int <= 0:
            return mask

        eroded = mask.astype(bool)

        # Erosió iterativa en 4 veïns
        for _ in range(margin_int):
            up = np.pad(eroded[:-1, :], ((1, 0), (0, 0)), constant_values=False)
            down = np.pad(eroded[1:, :], ((0, 1), (0, 0)), constant_values=False)
            left = np.pad(eroded[:, :-1], ((0, 0), (1, 0)), constant_values=False)
            right = np.pad(eroded[:, 1:], ((0, 0), (0, 1)), constant_values=False)

            eroded = up & down & left & right & eroded

        return eroded.astype(np.uint8)

    def get_mask_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Calcula el bounding box d'una màscara.
        Retorna (x1, y1, x2, y2) o None si està buida.
        """
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return None

        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        return x1, y1, x2, y2

    # -------------------------------------------------------------
    # 1) Normalitzar entrada i carregar imatge
    # -------------------------------------------------------------
    if not isinstance(image, (str, Path)):
        raise TypeError("`image` ha de ser una ruta (str o Path) a la imatge RGB.")

    image_path = str(image)
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size

    # -------------------------------------------------------------
    # 2) Inferència amb Roboflow
    # -------------------------------------------------------------
    api_key = os.environ["ROBOFLOW_API_KEY"]  # es dona per fet que ja està carregada
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    result = client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)

    # -------------------------------------------------------------
    # 3) Extreure màscares d'instància
    # -------------------------------------------------------------
    masks = extract_instance_masks_from_result(result, w, h)

    # Si no hi ha màscara → retornem None i la imatge original com a visualització
    if not masks:
        return None, pil_img

    # Combinar màscares en una sola màscara global
    global_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        global_mask |= (m > 0).astype(np.uint8)

    # -------------------------------------------------------------
    # 4) Erosió amb marge
    # -------------------------------------------------------------
    margin_int = int(margin) if margin is not None else 0
    eroded_mask = erode_mask(global_mask, margin_int)
    if eroded_mask.sum() == 0:
        # Si ens l'hem "menjat" tota, fem servir la global
        eroded_mask = global_mask

    # Contorns
    global_boundary = mask_boundary(global_mask)
    eroded_boundary = mask_boundary(eroded_mask)

    # Bounding box (primer intent amb la erosionada, sinó amb la global)
    bbox = get_mask_bounding_box(eroded_mask)
    if bbox is None:
        bbox = get_mask_bounding_box(global_mask)

    # -------------------------------------------------------------
    # 5) Imatge de visualització
    # -------------------------------------------------------------
    vis_img = pil_img.copy()
    draw = ImageDraw.Draw(vis_img)

    # Contorn original en vermell
    if global_boundary.any():
        ys, xs = np.where(global_boundary == 1)
        for x, y in zip(xs.tolist(), ys.tolist()):
            vis_img.putpixel((x, y), (255, 0, 0))

    # Contorn erosionat en verd
    if eroded_boundary.any():
        ys, xs = np.where(eroded_boundary == 1)
        for x, y in zip(xs.tolist(), ys.tolist()):
            vis_img.putpixel((x, y), (0, 255, 0))

    # Bounding box en blau
    if bbox is not None:
        draw.rectangle(bbox, outline="blue", width=2)

    # -------------------------------------------------------------
    # 6) Imatge cropped (amb fons negre fora màscara)
    # -------------------------------------------------------------
    cropped_img: Optional[Image.Image] = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img_array = np.array(pil_img)
        cropped_array = img_array[y1:y2, x1:x2].copy()
        cropped_mask = eroded_mask[y1:y2, x1:x2]

        # Fora de la màscara → negre
        cropped_array[cropped_mask == 0] = 0
        cropped_img = Image.fromarray(cropped_array)

    return cropped_img, vis_img


def potato_filter_extreme_colours(image: Image.Image, margin: int = 30, ignore_black: bool = True,
) -> Tuple[Image.Image, Tuple[float, float, float]]:
    """
    Filtra la imatge mantenint només els píxels propers al color RGB mediana.

    - Calcula la mediana R, G, B dels píxels.
    - Elimina (posa a negre) tots els píxels que no estiguin dins d'un marge
      de tolerància respecte aquesta mediana.
    - Opcionalment pot ignorar els píxels negres (0,0,0) per al càlcul de la mediana,
      útil quan la imatge ve d'un mask/crop on el fons és negre.

    Paràmetres
    ----------
    image : PIL.Image.Image
        Imatge RGB d'entrada.
    margin : int, opcional
        Màxim desviament permès per canal respecte la mediana.
        Si |canal - mediana_canal| > margin → es posa a negre.
    ignore_black : bool, opcional
        Si True, s'ignoren els píxels (0,0,0) a l'hora de calcular la mediana.

    Retorna
    -------
    filtered_img : PIL.Image.Image
        Imatge on només queden (sense canviar el color) els píxels
        propers a la mediana; la resta es posen a negre.
    median_color : tuple(float, float, float)
        Color mediana (R_med, G_med, B_med) utilitzat en el filtre.
    """

    if image.mode != "RGB":
        img_rgb = image.convert("RGB")
    else:
        img_rgb = image

    arr = np.array(img_rgb, dtype=np.uint8)  # (H, W, 3)

    # Aplanem per calcular la mediana
    flat = arr.reshape(-1, 3)

    if ignore_black:
        # Considerem només píxels que no són completament negres
        mask_non_black = ~( (flat[:, 0] == 0) & (flat[:, 1] == 0) & (flat[:, 2] == 0) )
        valid_pixels = flat[mask_non_black]
    else:
        valid_pixels = flat

    # Si no hi ha píxels vàlids, retornem la imatge tal qual i mediana (0,0,0)
    if valid_pixels.size == 0:
        median_color = (0.0, 0.0, 0.0)
        return img_rgb.copy(), median_color

    # Mediana per canal
    median_r = float(np.median(valid_pixels[:, 0]))
    median_g = float(np.median(valid_pixels[:, 1]))
    median_b = float(np.median(valid_pixels[:, 2]))
    median_color = (median_r, median_g, median_b)

    # Calculem la desviació absoluta per canal
    diff_r = np.abs(arr[:, :, 0].astype(np.int32) - median_r)
    diff_g = np.abs(arr[:, :, 1].astype(np.int32) - median_g)
    diff_b = np.abs(arr[:, :, 2].astype(np.int32) - median_b)

    # Condició de "proximitat": tots els canals dins del marge
    close_mask = (
        (diff_r <= margin) &
        (diff_g <= margin) &
        (diff_b <= margin)
    )

    # Creem una còpia de la imatge i posem a negre els píxels llunyans
    filtered_arr = arr.copy()
    filtered_arr[~close_mask] = 0  # negre

    filtered_img = Image.fromarray(filtered_arr, mode="RGB")
    return filtered_img


def nir_scalation(nir, reference_val):
    """
    Escala/normalitza el canal NIR de la imatge.

    Paràmetres
    ----------
    nir : Any
        Canal NIR.
    reference_val : Any
        Valor de referència per a l'escala/normalització.
    """
    pass