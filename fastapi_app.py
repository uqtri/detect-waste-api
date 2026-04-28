import asyncio
import os
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen

import cloudinary
import cloudinary.uploader
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import make_predictions_1 as predictor


load_dotenv()


DETECTOR_NAME = os.getenv("DETECTOR_NAME", "tf_efficientdet_d2")
DETECTOR_CHECKPOINT = os.getenv("DETECTOR_CHECKPOINT")
DEVICE = os.getenv("PREDICTION_DEVICE", "cpu")
PROB_THRESHOLD = float(os.getenv("PREDICTION_PROB_THRESHOLD", "0.17"))
CLS_THRESHOLD = float(os.getenv("PREDICTION_CLS_THRESHOLD", "0.5"))
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "detect-waste")
MAX_PARALLEL_IMAGES = max(1, int(os.getenv("MAX_PARALLEL_IMAGES", "4")))


app = FastAPI(title="Detect Waste API", version="1.0.0")


class PredictRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1)


def _configure_cloudinary() -> None:
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if not all([cloud_name, api_key, api_secret]):
        raise RuntimeError(
            "Cloudinary is not configured. Set CLOUDINARY_CLOUD_NAME, "
            "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET."
        )

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )


@lru_cache(maxsize=1)
def get_detector():
    if not DETECTOR_CHECKPOINT:
        raise RuntimeError(
            "DETECTOR_CHECKPOINT is not set. Provide a path to detector weights."
        )

    torch.set_grad_enabled(False)
    detector = predictor.set_model(DETECTOR_NAME, 1, DETECTOR_CHECKPOINT, DEVICE)
    detector.eval()
    return detector


def run_prediction(input_image_path: str, output_image_path: str) -> dict:
    detector = get_detector()
    image = predictor.Image.open(input_image_path).convert("RGB")

    img_tensor = predictor.get_transforms(image)
    outputs = detector(img_tensor.to(DEVICE))

    bboxes_keep = outputs[0, outputs[0, :, 4] > PROB_THRESHOLD]
    if bboxes_keep.numel() == 0:
        predictor.plot_results(image, [], [], predictor.classes, output_image_path)
        return {"count": 0, "boxes": []}

    probas = bboxes_keep[:, 4:]
    bboxes_scaled = predictor.rescale_bboxes(
        bboxes_keep[:, :4], image.size, tuple(img_tensor.size()[2:])
    )

    bboxes_final = []
    cls_prob = []
    kept_boxes = []

    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        crop = image.crop((xmin, ymin, xmax, ymax))
        cls_image = predictor.cls_transform(crop).unsqueeze(0)

        with torch.no_grad():
            cls_outputs = predictor.classifier(cls_image)

        probs = F.softmax(cls_outputs, dim=1)[0]
        pred = probs.argmax().item()
        conf = probs[pred].item()

        # plot_results expects class id to be 1-based.
        p[1] = pred + 1
        p[0] = conf

        if conf >= CLS_THRESHOLD:
            bboxes_final.append((xmin, ymin, xmax, ymax))
            cls_prob.append(p)
            kept_boxes.append(
                {
                    "label": predictor.classes[pred]
                    if 0 <= pred < len(predictor.classes)
                    else str(pred),
                    "class_id": pred,
                    "confidence": conf,
                    "bbox": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    },
                }
            )

    predictor.plot_results(
        image,
        cls_prob,
        bboxes_final,
        predictor.classes,
        output_image_path,
    )
    return {"count": len(bboxes_final), "boxes": kept_boxes}


def _download_image_from_url(image_url: str, dst_path: str) -> None:
    request = URLRequest(image_url, headers={"User-Agent": "detect-waste-api/1.0"})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise RuntimeError("image_url must point to an image resource.")
        data = response.read()

    if not data:
        raise RuntimeError("image_url returned empty content.")

    with open(dst_path, "wb") as dst:
        dst.write(data)


def _upload_prediction_to_cloudinary(image_path: str, public_id: str) -> str:
    result = cloudinary.uploader.upload(
        image_path,
        folder=CLOUDINARY_FOLDER,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
    )
    return result["secure_url"]


def _process_one_image(image_url: str) -> dict:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix or ".jpg"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_image_path = os.path.join(temp_dir, f"input{suffix}")
        output_image_path = os.path.join(temp_dir, "prediction.png")

        _download_image_from_url(image_url, input_image_path)
        prediction = run_prediction(input_image_path, output_image_path)
        predicted_url = _upload_prediction_to_cloudinary(
            output_image_path,
            public_id=f"prediction-{uuid.uuid4().hex}",
        )

    return {
        "source_url": image_url,
        "detections": prediction["count"],
        "predicted_url": predicted_url,
        "boxes": prediction["boxes"],
    }


async def _process_one_image_async(image_url: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        return await asyncio.to_thread(_process_one_image, image_url)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_images(payload: PredictRequest):
    if not payload.image_urls:
        raise HTTPException(status_code=400, detail="image_urls cannot be empty.")

    try:
        _configure_cloudinary()

        semaphore = asyncio.Semaphore(MAX_PARALLEL_IMAGES)
        tasks = [
            _process_one_image_async(image_url, semaphore)
            for image_url in payload.image_urls
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for image_url, item in zip(payload.image_urls, task_results):
            if isinstance(item, Exception):
                results.append({
                    "source_url": image_url,
                    "error": str(item),
                })
            else:
                results.append(item)

        urls = [
            item.get("predicted_url")
            for item in results
            if isinstance(item, dict) and item.get("predicted_url")
        ]

        # Backward/forward compatible payload:
        # - `urls`: convenience list of generated prediction image URLs
        # - `results`: per-image objects (source_url, predicted_url, detections, error, ...)
        return {"urls": urls, "results": results}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc