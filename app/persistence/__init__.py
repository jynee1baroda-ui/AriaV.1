"""Persistence layer: SQLAlchemy models and (from M2) the repository.

Kept separate from the domain models (app/models.py) on purpose:
the domain's Message/Conversation never import SQLAlchemy. This is
hexagonal architecture - swap the database and the domain doesn't
even know.
"""
