import asyncio
from pathlib import Path

from worker import celery_app
from config import settings
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review
from scripts.index_repo import index_directory
from db.session import AsyncSessionLocal
from db.models import IndexedFile, PRReview
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, timezone
import hashlib