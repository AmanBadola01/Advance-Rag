"""Request-local account identity; never accept account IDs from request payloads."""
from contextvars import ContextVar
account_id = ContextVar('account_id', default=None)
