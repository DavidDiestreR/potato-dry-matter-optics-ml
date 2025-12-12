"""
Mòdul: raw_image_treatment
Funcions relacionades amb el preprocés de la il·luminació (RGB i NIR).
"""

#from http import client
from typing import Tuple, Optional, Any, List, Dict, Union
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
from roboflow import Roboflow
from contextlib import redirect_stdout
import io
import tempfile


def apply_brightness_and_gamma(
    image: Union[str, Path, os.PathLike, Image.Image, np.ndarray],
    brightness: float = 2.0,
    gamma: float = 0.8,
) -> Image.Image:
    """
    Aplica brightness i correcció gamma a una imatge RGB.

    Paràmetres
    ----------
    image : str | pathlib.Path | PIL.Image.Image | np.ndarray
        Ruta a la imatge RGB al disc o objecte imatge (PIL / numpy).
    brightness : float
        Factor de brightness (per defecte 2.0).
    gamma : float
        Gamma (per defecte 0.8). (arr ** gamma)

    Retorna
    -------
    PIL.Image.Image
        Imatge RGB amb brightness i gamma aplicats.
    """

    def _to_pil_image(img_in) -> Image.Image:
        if isinstance(img_in, Image.Image):
            return img_in.convert("RGB")

        if isinstance(img_in, (str, Path, os.PathLike)):
            return Image.open(str(img_in)).convert("RGB")

        if isinstance(img_in, np.ndarray):
            arr = img_in
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"numpy array amb forma invàlida: {arr.shape}")

            if arr.shape[2] == 4:
                arr = arr[:, :, :3]

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return Image.fromarray(arr, mode="RGB")

        raise TypeError("`image` ha de ser str/Path, PIL.Image o numpy.ndarray")

    def _gamma_correction(pil_img: Image.Image, g: float) -> Image.Image:
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        arr = arr ** float(g)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    pil = _to_pil_image(image)

    # Brightness
    if brightness is not None:
        pil = ImageEnhance.Brightness(pil).enhance(float(brightness))

    # Gamma
    if gamma is not None:
        pil = _gamma_correction(pil, float(gamma))

    return pil


def potato_defect_classification(image: Any, confidence_threshold: float = 0.40
) -> Tuple[str, float, Image.Image]:
    """
    Classifica el defecte de la patata a partir d'una imatge amb Roboflow i genera
    una imatge de visualització amb bounding box i etiqueta.

    Paràmetres
    ----------
    image : str | pathlib.Path | PIL.Image.Image | np.ndarray
        Ruta a la imatge RGB al disc o objecte imatge (PIL / numpy).
    confidence_threshold : float, opcional
        Llindar mínim de confiança per acceptar una predicció. Per defecte 0.40.

    Retorna
    -------
    defect : str
        Nom del defecte si hi ha detecció >= threshold, si no "Unable to classify".
    confidence : float
        Confiança de la millor predicció. Si no hi ha predicció vàlida, 0.0.
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
        """Extreu bbox del format típic Roboflow."""
        if all(k in pred for k in ("x", "y", "width", "height")):
            x = float(pred["x"])
            y = float(pred["y"])
            w = float(pred["width"])
            h = float(pred["height"])
            return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

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

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        pad = 4
        if font is not None and hasattr(draw, "textbbox"):
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = int(draw.textlength(label)), 12

        if bbox is not None:
            tx = max(0, int(x1))
            ty = max(0, int(y1) - th - 2 * pad)
        else:
            tx, ty = 0, 0

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
        return "Unable to classify", 0.0, pil_img

    vis_img = _draw_annotation(pil_img, best, float(confidence_threshold))
    defect = str(best.get("class", "unknown"))
    conf = float(best.get("confidence", 0.0))
    return defect, conf, vis_img


def potato_pixels_rgb_img(image: Any, margin: int = 0, min_conf: float = 0.01
) -> Tuple[Optional[Image.Image], Image.Image]:
    """
    Preprocessa la imatge en el rang visible (RGB) utilitzant un model de Roboflow.

    El procediment és el següent:
    - Segmenta la patata mitjançant *instance segmentation*.
    - Filtra les deteccions segons la confidence i selecciona la més fiable.
    - Erosiona la màscara amb un marge cap a dins.
    - Genera una imatge de visualització amb el contorn original, el contorn erosionat
      i la *bounding box*.
    - Genera una imatge retallada (*cropped*) de la patata, amb fons negre fora de la
      màscara.

    Paràmetres
    ----------
    image : str | pathlib.Path | PIL.Image.Image | np.ndarray
        Ruta a la imatge RGB al disc o objecte imatge (PIL o numpy).
    margin : int, opcional
        Mida de l'erosió cap a dins, en píxels. Per defecte és 0.
    min_conf : float, opcional
        Valor mínim de *confidence* (entre 0 i 1) per acceptar una detecció.
        Per defecte és 0.01.

    Retorna
    -------
    cropped_img : PIL.Image.Image o None
        Imatge retallada de la patata, amb el fons negre fora de la màscara.
        Retorna None si no es detecta cap patata.
    vis_img : PIL.Image.Image
        Imatge original amb els contorns i la *bounding box* dibuixats.
    """

    tmp_path = None  # path temporal (PNG RGB) només si cal

    # -------------------------------------------------------------
    # 0) Carregar ORIGINAL -> PIL i eliminar alfa (treballarem sempre en RGB)
    # -------------------------------------------------------------
    if isinstance(image, (str, Path, os.PathLike)):
        orig_path = str(image)
        pil_orig = Image.open(orig_path)
    elif isinstance(image, Image.Image):
        orig_path = None
        pil_orig = image
    elif isinstance(image, np.ndarray):
        orig_path = None
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise TypeError(f"np.ndarray amb forma invàlida: {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_orig = Image.fromarray(arr)
    else:
        raise TypeError("image ha de ser path, PIL.Image o np.ndarray")

    # elimina alfa / modes estranys
    if pil_orig.mode != "RGB":
        pil_orig = pil_orig.convert("RGB")

    w, h = pil_orig.size

    # -------------------------------------------------------------
    # 1) Decidir infer_path (SEMPRE un PNG RGB si hi ha risc de RGBA)
    # -------------------------------------------------------------
    if orig_path is not None:
        # IMPORTANT: encara que nosaltres hàgim convertit pil_orig,
        # si passem orig_path, Roboflow re-obrirà el fitxer original (pot ser RGBA).
        # Per tant, si el fitxer original NO és RGB, creem un PNG RGB temporal.
        with Image.open(orig_path) as chk:
            needs_sanitize = (chk.mode != "RGB")

        if needs_sanitize:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil_orig.save(tmp_path, format="PNG", optimize=False)
            infer_path = tmp_path
        else:
            infer_path = orig_path
    else:
        # no tenim path: forcem PNG temporal RGB
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pil_orig.save(tmp_path, format="PNG", optimize=False)
        infer_path = tmp_path

    try:
        # -------------------------------------------------------------
        # 2) Inferència Roboflow (path)
        # -------------------------------------------------------------
        with redirect_stdout(io.StringIO()):
            rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
            project = rf.workspace("microsoft").project("coco-dataset-vdnr1")
            model = project.version(23).model

        conf_pct = int(min_conf * 100)

        result = model.predict(infer_path, confidence=conf_pct).json()
        preds = result.get("predictions", [])
        if not preds:
            return None, pil_orig

        # -------------------------------------------------------------
        # 3) Màscara del millor prediction (mateix criteri)
        # -------------------------------------------------------------
        best_pred = max(preds, key=lambda p: p["confidence"])
        points = best_pred.get("points")
        if not points or len(points) < 3:
            return None, pil_orig

        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1)

        # -------------------------------------------------------------
        # 4) Erosió
        # -------------------------------------------------------------
        if margin > 0:
            kernel = np.ones((3, 3), np.uint8)
            eroded = mask.copy()
            for _ in range(int(margin)):
                eroded = cv2.erode(eroded, kernel)
            mask_eroded = eroded if eroded.sum() > 0 else mask
        else:
            mask_eroded = mask

        # -------------------------------------------------------------
        # 5) Bounding box
        # -------------------------------------------------------------
        ys, xs = np.where(mask_eroded > 0)
        if len(xs) == 0:
            return None, pil_orig

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        # -------------------------------------------------------------
        # 6) Visualització (RGB)
        # -------------------------------------------------------------
        vis_np = np.array(pil_orig).copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_np, contours, -1, (255, 0, 0), 2)

        contours_e, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_np, contours_e, -1, (0, 255, 0), 2)

        cv2.rectangle(vis_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
        vis_img = Image.fromarray(vis_np, mode="RGB")

        # -------------------------------------------------------------
        # 7) Crop sobre la PNG original (ja en RGB, sense alfa)
        # -------------------------------------------------------------
        img_np = np.array(pil_orig)
        cropped = img_np[y1:y2, x1:x2].copy()
        cropped_mask = mask_eroded[y1:y2, x1:x2]
        cropped[cropped_mask == 0] = 0
        cropped_img = Image.fromarray(cropped, mode="RGB")

        return cropped_img, vis_img

    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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
    Escala/normalitza el canal NIR dividint-lo pel valor de referència.

    - Si nir i reference_val són vectors/arrays: divisió element a element.
    - Si són escalars: divisió escalar.

    Paràmetres
    ----------
    nir : Any
        Canal NIR (escalar, llista, numpy array).
    reference_val : Any
        Valor(s) de referència per a l'escala (mateixa forma que nir o escalar).
    """
    nir_arr = np.asarray(nir, dtype=np.float32)
    ref_arr = np.asarray(reference_val, dtype=np.float32)

    # Si tots dos són "vectorials" (ndim > 0), comprovem forma
    if nir_arr.ndim > 0 and ref_arr.ndim > 0:
        if nir_arr.shape != ref_arr.shape:
            raise ValueError(f"nir i reference_val han de tenir la mateixa forma. "
                             f"Got nir {nir_arr.shape} vs ref {ref_arr.shape}")

    # Evitar divisions per zero
    if np.any(ref_arr == 0):
        raise ZeroDivisionError("reference_val conté zeros; no es pot dividir per zero.")

    out = nir_arr / ref_arr

    # Retornem escalar "net" si l'entrada era escalar
    if out.ndim == 0:
        return float(out)
    return out