"""
CropCare AI - Database Initialization
Initialize database with tables and optional seed data.
"""

from src.database.models import init_database, drop_database, get_engine
from src.database.models import Base
from src.config import DATABASE_PATH


def initialize_database(reset: bool = False):
    """
    Initialize the database.

    Args:
        reset: If True, drop existing tables and recreate
    """
    if reset:
        print("Resetting database...")
        drop_database()

    print("Initializing database...")
    init_database()

    # Verify tables created
    engine = get_engine()
    inspector = __import__("sqlalchemy").inspect(engine)
    tables = inspector.get_table_names()
    print(f"Created tables: {tables}")

    print(f"Database ready at: {DATABASE_PATH}")


def verify_database() -> bool:
    """Verify database is accessible and has correct schema."""
    try:
        engine = get_engine()
        inspector = __import__("sqlalchemy").inspect(engine)
        tables = inspector.get_table_names()

        required_tables = ["predictions", "model_metrics"]
        for table in required_tables:
            if table not in tables:
                print(f"Missing table: {table}")
                return False

        # Check predictions table columns
        columns = [col["name"] for col in inspector.get_columns("predictions")]
        required_columns = [
            "id", "timestamp", "crop", "disease", "confidence",
            "health_status", "image_path", "gradcam_path", "language", "notes"
        ]
        for col in required_columns:
            if col not in columns:
                print(f"Missing column in predictions: {col}")
                return False

        print("Database verification passed!")
        return True
    except Exception as e:
        print(f"Database verification failed: {e}")
        return False


def backup_database(backup_path: str):
    """Create a backup of the database."""
    import shutil
    shutil.copy2(DATABASE_PATH, backup_path)
    print(f"Database backed up to: {backup_path}")


def restore_database(backup_path: str):
    """Restore database from backup."""
    import shutil
    shutil.copy2(backup_path, DATABASE_PATH)
    print(f"Database restored from: {backup_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        initialize_database(reset=True)
    else:
        initialize_database()