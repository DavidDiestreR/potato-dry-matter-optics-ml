"""
Mòdul: raw_image_treatment
Funcions relacionades amb el preprocés de la il·luminació (RGB i NIR).
"""

from http import client
from typing import Tuple, Optional, Any, List, Dict
from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient


def potato_defect_classification(image: Any, confidence_threshold: float = 0.40) -> Tuple[object, Image.Image]:
    """
    Classifica el defecte de la patata a partir d'una imatge amb Roboflow i genera
    una imatge de visualització amb bounding box i etiqueta.

    Paràmetres
    ----------
    image : Any
        Ruta (str/Path), PIL.Image o numpy array.
    confidence_threshold : float
        Llindar mínim de confiança per acceptar una predicció.

    Retorna
    -------
    defect : object
        Nom del defecte (str) si hi ha detecció >= threshold, si no np.nan.
    vis_img : PIL.Image.Image
        Imatge anotada (bbox + text). Si no detecta res, imatge original.
    """

    ROBOFLOW_MODEL_ID = "potato-detection-3et6q/11"

    # ------------------------------------------------------------------
    # Helpers interns
    # ------------------------------------------------------------------

    def _to_pil_image(img_in: Any) -> Image.Image:
        """Normalitza input (path/PIL/numpy) a PIL.Image RGB."""
        if isinstance(img_in, Image.Image):
            return img_in.convert("RGB")

        if isinstance(img_in, (str, Path, os.PathLike)):
            return Image.open(str(img_in)).convert("RGB")

        if isinstance(img_in, np.ndarray):
            arr = img_in
            if arr.ndim == 2:  # gris -> RGB
                arr = np.stack([arr, arr, arr], axis=-1)

            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"numpy array amb forma invàlida: {arr.shape}")

            if arr.shape[2] == 4:  # RGBA -> RGB
                arr = arr[:, :, :3]

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return Image.fromarray(arr, mode="RGB")

        raise TypeError("`image` ha de ser path (str/Path), PIL.Image o numpy.ndarray")

    def _infer(pil_img: Image.Image) -> Dict[str, Any]:
        """Crida Roboflow passant la imatge com PIL.Image (sense bytes)."""
        api_key = os.environ["ROBOFLOW_API_KEY"]  # es dona per fet que ja està carregada
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

        return client.infer(pil_img, model_id=ROBOFLOW_MODEL_ID)


    def _pick_best_prediction(result: Dict[str, Any], thr: float) -> Optional[Dict[str, Any]]:
        """Tria la millor predicció per confiança, si supera thr."""
        preds: List[Dict[str, Any]] = result.get("predictions", []) if isinstance(result, dict) else []
        if not preds:
            return None

        best = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
        if float(best.get("confidence", 0.0)) < thr:
            return None

        return best

    def _bbox_from_pred(pred: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """
        Extreu bbox del format típic Roboflow:
        - (x, y, width, height) centrats
        O bé:
        - (x1, y1, x2, y2) si vingués així
        """
        if all(k in pred for k in ("x", "y", "width", "height")):
            x = float(pred["x"])
            y = float(pred["y"])
            w = float(pred["width"])
            h = float(pred["height"])
            return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

        # fallback per si el teu model retorna cantonades
        if all(k in pred for k in ("x1", "y1", "x2", "y2")):
            return (float(pred["x1"]), float(pred["y1"]), float(pred["x2"]), float(pred["y2"]))

        return None

    def _draw_annotation(base_img: Image.Image, pred: Dict[str, Any], thr: float) -> Image.Image:
        """Dibuixa bbox i label a una còpia de la imatge."""
        out = base_img.copy()
        draw = ImageDraw.Draw(out)

        bbox = _bbox_from_pred(pred)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], width=3)

        cls = str(pred.get("class", "unknown"))
        conf = float(pred.get("confidence", 0.0))
        label = f"{cls} | conf={conf:.2f} | thr={thr:.2f}"

        # font (default)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # mida label
        pad = 4
        if font is not None and hasattr(draw, "textbbox"):
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = int(draw.textlength(label)), 12  # fallback

        # posició label (a sobre del bbox si hi ha, sinó a dalt-esquerra)
        if bbox is not None:
            tx = max(0, int(x1))
            ty = max(0, int(y1) - th - 2 * pad)
        else:
            tx, ty = 0, 0

        # caixa negra + text blanc
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
        draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255), font=font)

        return out

    # ------------------------------------------------------------------
    # Flux principal
    # ------------------------------------------------------------------
    pil_img = _to_pil_image(image)
    result = _infer(pil_img)

    best = _pick_best_prediction(result, float(confidence_threshold))
    if best is None:
        return np.nan, pil_img

    vis_img = _draw_annotation(pil_img, best, float(confidence_threshold))
    defect = str(best.get("class", "unknown"))
    return defect, vis_img


def potato_pixels_rgb_img(image: Any, margin: int = 0) -> Tuple[Optional[Image.Image], Image.Image]:
    """
    Preprocessa la imatge en el rang visible (RGB) amb el model de Roboflow:
    - Segmenta la patata.
    - Erosiona la màscara amb un marge cap a dins.
    - Genera una imatge de visualització (contorn original + erosionat + bbox).
    - Genera una imatge 'cropped' de la patata (fons negre fora de la màscara).

    Paràmetres
    ----------
    image : str | pathlib.Path | PIL.Image.Image | np.ndarray
        Ruta a la imatge RGB al disc o objecte imatge (PIL / numpy).
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
        pts = (
            pred.get("points")
            or pred.get("polygon")
            or pred.get("vertices")
            or pred.get("segmentation")
        )
        if not pts:
            return None

        if isinstance(pts, list) and len(pts) > 0:
            if isinstance(pts[0], dict):
                out = []
                for p in pts:
                    x = p.get("x")
                    y = p.get("y")
                    if x is not None and y is not None:
                        out.append((float(x), float(y)))
                return out if len(out) >= 3 else None

            if isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
                out = [(float(p[0]), float(p[1])) for p in pts]
                return out if len(out) >= 3 else None

        return None

    def polygon_to_mask(points: List[Tuple[float, float]], w: int, h: int) -> np.ndarray:
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        pts_int = [(int(x), int(y)) for x, y in points]
        draw.polygon(pts_int, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _normalize_mask(m: Any, w: int, h: int) -> Optional[np.ndarray]:
        if m is None:
            return None

        if isinstance(m, np.ndarray):
            arr = m
        elif isinstance(m, (list, tuple)):
            arr = np.array(m)
        elif isinstance(m, dict) and "data" in m:
            arr = np.array(m["data"])
        else:
            try:
                arr = np.array(m)
            except Exception:
                return None

        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] == 1:
                arr = arr[0]

        if arr.ndim != 2:
            return None

        arr = (arr > 0).astype(np.uint8)

        if arr.shape != (h, w):
            if arr.size == h * w:
                arr = arr.reshape((h, w))
            else:
                mask_img = Image.fromarray(arr * 255)
                mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
                arr = (np.array(mask_img) > 0).astype(np.uint8)

        return arr

    def extract_instance_masks_from_result(result: Dict[str, Any], w: int, h: int) -> List[np.ndarray]:
        masks_local: List[np.ndarray] = []
        preds = result.get("predictions", []) if isinstance(result, dict) else []

        for pred in preds:
            points = _prediction_to_points(pred)
            if points:
                masks_local.append(polygon_to_mask(points, w, h))

        if masks_local:
            return masks_local

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
        m = mask.astype(bool)

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
        if margin_int <= 0:
            return mask

        eroded = mask.astype(bool)

        for _ in range(margin_int):
            up = np.pad(eroded[:-1, :], ((1, 0), (0, 0)), constant_values=False)
            down = np.pad(eroded[1:, :], ((0, 1), (0, 0)), constant_values=False)
            left = np.pad(eroded[:, :-1], ((0, 0), (1, 0)), constant_values=False)
            right = np.pad(eroded[:, 1:], ((0, 0), (0, 1)), constant_values=False)
            eroded = up & down & left & right & eroded

        return eroded.astype(np.uint8)

    def get_mask_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return None
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        return x1, y1, x2, y2

    def _to_pil_rgb(img_in: Any) -> Tuple[Image.Image, Optional[str]]:
        """
        Retorna (pil_img_RGB, image_path_if_any).
        Si és ruta, retorna path; si és objecte, retorna None.
        """
        if isinstance(img_in, (str, Path, os.PathLike)):
            p = str(img_in)
            return Image.open(p).convert("RGB"), p

        if isinstance(img_in, Image.Image):
            return img_in.convert("RGB"), None

        if isinstance(img_in, np.ndarray):
            arr = img_in
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise TypeError(f"np.ndarray amb forma invàlida: {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, "RGB"), None

        raise TypeError("`image` ha de ser str/Path, PIL.Image o np.ndarray.")

    def _infer(client: InferenceHTTPClient, pil_img: Image.Image, image_path: Optional[str]) -> Dict[str, Any]:
        if image_path is not None:
            return client.infer(image_path, model_id=ROBOFLOW_MODEL_ID)

        return client.infer(pil_img, model_id=ROBOFLOW_MODEL_ID)

    # -------------------------------------------------------------
    # 1) Normalitzar entrada i carregar imatge
    # -------------------------------------------------------------
    pil_img, image_path = _to_pil_rgb(image)
    w, h = pil_img.size

    # -------------------------------------------------------------
    # 2) Inferència amb Roboflow
    # -------------------------------------------------------------
    api_key = os.environ["ROBOFLOW_API_KEY"]  # es dona per fet que ja està carregada
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    result = _infer(client, pil_img, image_path)

    # -------------------------------------------------------------
    # 3) Extreure màscares d'instància
    # -------------------------------------------------------------
    masks = extract_instance_masks_from_result(result, w, h)

    if not masks:
        return None, pil_img

    global_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        global_mask |= (m > 0).astype(np.uint8)

    # -------------------------------------------------------------
    # 4) Erosió amb marge
    # -------------------------------------------------------------
    margin_int = int(margin) if margin is not None else 0
    eroded_mask = erode_mask(global_mask, margin_int)
    if eroded_mask.sum() == 0:
        eroded_mask = global_mask

    global_boundary = mask_boundary(global_mask)
    eroded_boundary = mask_boundary(eroded_mask)

    bbox = get_mask_bounding_box(eroded_mask)
    if bbox is None:
        bbox = get_mask_bounding_box(global_mask)

    # -------------------------------------------------------------
    # 5) Imatge de visualització
    # -------------------------------------------------------------
    vis_img = pil_img.copy()
    draw = ImageDraw.Draw(vis_img)

    if global_boundary.any():
        ys, xs = np.where(global_boundary == 1)
        for x, y in zip(xs.tolist(), ys.tolist()):
            vis_img.putpixel((x, y), (255, 0, 0))

    if eroded_boundary.any():
        ys, xs = np.where(eroded_boundary == 1)
        for x, y in zip(xs.tolist(), ys.tolist()):
            vis_img.putpixel((x, y), (0, 255, 0))

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