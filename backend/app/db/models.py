import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.database import Base

class DatasetEntity(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=False)
    col_count = Column(Integer, nullable=False)
    target_column = Column(String, nullable=True)
    id_column = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    experiments = relationship("ExperimentEntity", back_populates="dataset", cascade="all, delete-orphan")

class ExperimentEntity(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    target_column = Column(String, nullable=False)
    problem_type = Column(String, nullable=False) # "classification" or "regression"
    test_size = Column(Float, default=0.2)
    random_state = Column(Integer, default=42)
    preprocessing_config = Column(JSON, nullable=True)
    training_config = Column(JSON, nullable=True)
    status = Column(String, default="configured") # configured, training, completed, failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    dataset = relationship("DatasetEntity", back_populates="experiments")
    model_runs = relationship("ModelRunEntity", back_populates="experiment", cascade="all, delete-orphan")

class ModelRunEntity(Base):
    __tablename__ = "model_runs"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False) # classification / regression
    status = Column(String, default="queued") # queued, training, completed, failed
    training_time_sec = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    cv_scores = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    artifact_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    experiment = relationship("ExperimentEntity", back_populates="model_runs")
    predictions = relationship("PredictionHistoryEntity", back_populates="model_run", cascade="all, delete-orphan")

class PredictionHistoryEntity(Base):
    __tablename__ = "prediction_history"

    id = Column(String, primary_key=True, index=True)
    model_run_id = Column(String, ForeignKey("model_runs.id"), nullable=False)
    input_data = Column(JSON, nullable=False)
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=True)
    explanation = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    model_run = relationship("ModelRunEntity", back_populates="predictions")
