import logging
from functools import lru_cache
from typing import Optional

import spacy

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_spacy_model() -> Optional["spacy.language.Language"]:
    """Load and cache the best available spaCy English model in-process.

    Order of preference: en_core_web_lg, en_core_web_md, en_core_web_sm.
    Returns None if no model is available.
    """
    model_preferences = [
        "en_core_web_lg",
        "en_core_web_md",
        "en_core_web_sm",
    ]

    for model_name in model_preferences:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            logger.debug(f"spaCy model not available: {model_name}")
            continue

    logger.warning("No spaCy model found. Install one with: python -m spacy download en_core_web_md")
    return None


