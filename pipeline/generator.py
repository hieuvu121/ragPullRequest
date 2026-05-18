import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.generator import generate_review, ReviewComment
from pipeline.retriever import ScoredChunk