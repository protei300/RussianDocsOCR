"""Data access, as plain functions taking the store as their first argument.

No classes, no service layer — the reference project's shape. The important
property is that **these signatures are the migration contract**: swapping the
filesystem store for SQLAlchemy replaces the bodies and nothing else.
``tests/service/test_repository_contract.py`` is what enforces that.
"""
