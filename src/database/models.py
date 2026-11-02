"""
CropCare AI - Database Models
SQLAlchemy models for prediction history storage.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from src.config import DATABASE_URL, DATABASE_PATH

Base = declarative_base()


class Prediction(Base):
    """Prediction history model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    crop = Column(String(50), nullable=False, index=True)
    disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    health_status = Column(String(20), nullable=False, index=True)  # Healthy/Diseased
    image_path = Column(String(500), nullable=True)
    gradcam_path = Column(String(500), nullable=True)
    language = Column(String(10), default="en", nullable=False)
    notes = Column(Text, nullable=True)

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_crop_timestamp", "crop", "timestamp"),
        Index("ix_health_timestamp", "health_status", "timestamp"),
        Index("ix_disease_timestamp", "disease", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "crop": self.crop,
            "disease": self.disease,
            "confidence": self.confidence,
            "health_status": self.health_status,
            "image_path": self.image_path,
            "gradcam_path": self.gradcam_path,
            "language": self.language,
            "notes": self.notes,
        }

    def __repr__(self):
        return f"<Prediction(id={self.id}, crop={self.crop}, disease={self.disease}, confidence={self.confidence:.2f})>"


class ModelMetrics(Base):
    """Model evaluation metrics storage."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(50), nullable=False)
    dataset_version = Column(String(50), nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    test_samples = Column(Integer, nullable=True)
    metrics_json = Column(Text, nullable=True)  # Full metrics as JSON

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "test_samples": self.test_samples,
        }
        if self.metrics_json:
            try:
                result["metrics"] = json.loads(self.metrics_json)
            except Exception:
                result["metrics"] = None
        return result


# Database engine and session
_engine = None
_SessionFactory = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        # Ensure database directory exists
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DATABASE_URL, echo=False)
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


def get_session() -> Session:
    """Get a new database session."""
    return get_session_factory()()


def init_database():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Database initialized at {DATABASE_PATH}")


def drop_database():
    """Drop all tables (use with caution)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("Database tables dropped")


if __name__ == "__main__":
    init_database()