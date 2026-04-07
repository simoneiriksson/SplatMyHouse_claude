import os

BASE_URL = "https://api.dataforsyningen.dk/rest/skraafoto_api/v1.0"
DAWA_URL = "https://api.dataforsyningen.dk/adresser"
DEFAULT_COLLECTION = "skraafotos2023"

MIN_BASELINE = 50.0    # metres
MAX_BASELINE = 1500.0  # metres — wider to include oblique↔nadir pairs
MAX_PAIRS = 20
MAX_IMAGES = 30
SCENE_RADIUS = 300.0   # metres — XY crop radius around the target after reconstruction


def get_token(override: str | None = None) -> str:
    if override:
        return override
    token = os.environ.get("DATAFORSYNINGEN_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "No API token found. Set DATAFORSYNINGEN_TOKEN environment variable "
            "or pass --token."
        )
    return token
