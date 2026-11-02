"""
CropCare AI - Database CRUD Operations
Create, Read, Update, Delete operations for predictions.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.orm import Session

from src.database.models import (
    Prediction,
    ModelMetrics,
    get_session,
    init_database,
)


def create_prediction(
    crop: str,
    disease: str,
    confidence: float,
    health_status: str,
    image_path: Optional[str] = None,
    gradcam_path: Optional[str] = None,
    language: str = "en",
    notes: Optional[str] = None,
) -> Prediction:
    """Create a new prediction record."""
    session = get_session()
    try:
        prediction = Prediction(
            crop=crop,
            disease=disease,
            confidence=confidence,
            health_status=health_status,
            image_path=image_path,
            gradcam_path=gradcam_path,
            language=language,
            notes=notes,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_prediction(prediction_id: int) -> Optional[Prediction]:
    """Get a single prediction by ID."""
    session = get_session()
    try:
        return session.query(Prediction).filter(Prediction.id == prediction_id).first()
    finally:
        session.close()


def get_predictions(
    limit: int = 100,
    offset: int = 0,
    crop: Optional[str] = None,
    health_status: Optional[str] = None,
    disease: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    language: Optional[str] = None,
    order_by: str = "timestamp_desc",
) -> Tuple[List[Prediction], int]:
    """
    Get predictions with filtering and pagination.

    Returns:
        Tuple of (predictions_list, total_count)
    """
    session = get_session()
    try:
        query = session.query(Prediction)

        # Apply filters
        if crop:
            query = query.filter(Prediction.crop == crop)
        if health_status:
            query = query.filter(Prediction.health_status == health_status)
        if disease:
            query = query.filter(Prediction.disease == disease)
        if start_date:
            query = query.filter(Prediction.timestamp >= start_date)
        if end_date:
            query = query.filter(Prediction.timestamp <= end_date)
        if min_confidence is not None:
            query = query.filter(Prediction.confidence >= min_confidence)
        if max_confidence is not None:
            query = query.filter(Prediction.confidence <= max_confidence)
        if language:
            query = query.filter(Prediction.language == language)

        # Get total count before pagination
        total_count = query.count()

        # Apply ordering
        if order_by == "timestamp_desc":
            query = query.order_by(desc(Prediction.timestamp))
        elif order_by == "timestamp_asc":
            query = query.order_by(Prediction.timestamp)
        elif order_by == "confidence_desc":
            query = query.order_by(desc(Prediction.confidence))
        elif order_by == "confidence_asc":
            query = query.order_by(Prediction.confidence)

        # Apply pagination
        predictions = query.offset(offset).limit(limit).all()

        return predictions, total_count
    finally:
        session.close()


def delete_prediction(prediction_id: int) -> bool:
    """Delete a prediction by ID."""
    session = get_session()
    try:
        prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            session.delete(prediction)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def clear_all_predictions(confirm: bool = False) -> int:
    """Clear all prediction records. Requires confirm=True."""
    if not confirm:
        raise ValueError("Must confirm deletion with confirm=True")
    session = get_session()
    try:
        count = session.query(Prediction).count()
        session.query(Prediction).delete()
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_prediction_stats() -> Dict[str, Any]:
    """Get statistics about predictions."""
    session = get_session()
    try:
        total = session.query(Prediction).count()

        # By crop
        crop_stats = session.query(
            Prediction.crop,
            func.count(Prediction.id).label("count")
        ).group_by(Prediction.crop).all()

        # By health status
        health_stats = session.query(
            Prediction.health_status,
            func.count(Prediction.id).label("count")
        ).group_by(Prediction.health_status).all()

        # By disease
        disease_stats = session.query(
            Prediction.disease,
            func.count(Prediction.id).label("count")
        ).group_by(Prediction.disease).order_by(desc("count")).limit(10).all()

        # Average confidence
        avg_confidence = session.query(func.avg(Prediction.confidence)).scalar()

        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = session.query(Prediction).filter(
            Prediction.timestamp >= week_ago
        ).count()

        return {
            "total_predictions": total,
            "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
            "recent_predictions_7d": recent_count,
            "by_crop": {crop: count for crop, count in crop_stats},
            "by_health_status": {status: count for status, count in health_stats},
            "top_diseases": {disease: count for disease, count in disease_stats},
        }
    finally:
        session.close()


def search_predictions(
    query: str,
    limit: int = 50,
) -> List[Prediction]:
    """Search predictions by crop, disease, or notes."""
    session = get_session()
    try:
        search_term = f"%{query}%"
        predictions = session.query(Prediction).filter(
            or_(
                Prediction.crop.ilike(search_term),
                Prediction.disease.ilike(search_term),
                Prediction.notes.ilike(search_term),
            )
        ).order_by(desc(Prediction.timestamp)).limit(limit).all()
        return predictions
    finally:
        session.close()


def get_recent_predictions(limit: int = 10) -> List[Prediction]:
    """Get most recent predictions."""
    session = get_session()
    try:
        return session.query(Prediction).order_by(
            desc(Prediction.timestamp)
        ).limit(limit).all()
    finally:
        session.close()


def save_model_metrics(
    model_version: str,
    dataset_version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    test_samples: int,
    metrics_json: Optional[str] = None,
) -> ModelMetrics:
    """Save model evaluation metrics."""
    session = get_session()
    try:
        metrics = ModelMetrics(
            model_version=model_version,
            dataset_version=dataset_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            test_samples=test_samples,
            metrics_json=metrics_json,
        )
        session.add(metrics)
        session.commit()
        session.refresh(metrics)
        return metrics
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_latest_model_metrics() -> Optional[ModelMetrics]:
    """Get the most recent model metrics."""
    session = get_session()
    try:
        return session.query(ModelMetrics).order_by(
            desc(ModelMetrics.timestamp)
        ).first()
    finally:
        session.close()


if __name__ == "__main__":
    # Test CRUD operations
    init_database()

    # Create test prediction
    pred = create_prediction(
        crop="Tomato",
        disease="Tomato_Early_blight",
        confidence=0.92,
        health_status="Diseased",
        image_path="/path/to/image.jpg",
    )
    print(f"Created prediction: {pred.id}")

    # Get predictions
    preds, total = get_predictions(limit=5)
    print(f"Total: {total}, Retrieved: {len(preds)}")

    # Get stats
    stats = get_prediction_stats()
    print(f"Stats: {stats}")

    # Clean up test
    delete_prediction(pred.id)
    print("CRUD test passed!")