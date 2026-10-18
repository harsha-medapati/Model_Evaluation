from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import engine, Base
import app.db.models # Ensure models are loaded
from app.api.v1 import datasets, preprocessing, models, explain, predict, exports

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Professional ML Model Evaluation & Explainability Studio API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(preprocessing.router, prefix="/api/v1/preprocessing", tags=["Preprocessing"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["Explainability"])
app.include_router(predict.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(exports.router, prefix="/api/v1/exports", tags=["Exports"])

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "api_v1_docs": "/docs"
    }

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
