from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "ML Model Evaluation & Explainability Studio"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    SAMPLE_DIR: Path = DATA_DIR / "samples"
    ARTIFACT_DIR: Path = DATA_DIR / "artifacts"
    REPORT_DIR: Path = DATA_DIR / "reports"
    
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/studio.db"
    
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set = {".csv", ".xlsx", ".xls"}

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.UPLOAD_DIR, settings.SAMPLE_DIR, settings.ARTIFACT_DIR, settings.REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
