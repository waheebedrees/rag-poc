import os
import uuid
from pathlib import Path
from app.config import settings


def safe_filename(original: str, user_id: str) -> str:
    """Never trust user-supplied filenames in filesystem paths."""
    ext = Path(original).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Extension '.{ext}' not allowed. "
            f"Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )
    return f"{user_id}_{uuid.uuid4().hex}.{ext}"


def upload_path(filename: str) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    return os.path.join(settings.UPLOAD_DIR, filename)


def remove_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
