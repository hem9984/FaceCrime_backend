# services/face_rec.py
import os
import io
import base64
import logging
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Choose your recognition model here:
# options: 'arcface_r100_v1', 'adaface_ir101_webface12m', or others installed by insightface
RECOGNITION_MODEL = os.environ.get("RECOGNITION_MODEL", "buffalo_l")
# Detector size (will resize input or pad to maintain aspect)
DET_SIZE = (640, 640)

class FaceRecService:
    def __init__(self):
        logging.info(f"Initializing FaceRecService with model='{RECOGNITION_MODEL}' on GPU")
        # ctx_id=0 => first GPU, or -1 for CPU
        self.app = FaceAnalysis(
            name=RECOGNITION_MODEL,
            providers=['TensorrtExecutionProvider','CUDAExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)
        logging.info("FaceRecService ready")

    def extract(self, b64: str) -> list[float]:
        """
        Decode the base64-encoded image, detect the first face,
        compute and return its normalized embedding vector.
        """
        try:
            # strip off "data:image/..." prefix if present
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img_np = np.asarray(img)
            # detect & get embeddings
            faces = self.app.get(img_np)
            if not faces:
                logger.warning("No faces detected by InsightFace")
                return []
            # faces[0].embedding is a 512-d or 512-d normalized vector
            emb = faces[0].embedding.astype(float).tolist()
            return emb
        except Exception as e:
            logger.error(f"Failed to extract face embedding: {e}")
            return []

# singleton instance
_face_rec_svc = None

def extract_face_embedding(base64_image: str) -> list[float]:
    """
    Public function your routes will call.
    """
    global _face_rec_svc
    if _face_rec_svc is None:
        _face_rec_svc = FaceRecService()
    return _face_rec_svc.extract(base64_image)

