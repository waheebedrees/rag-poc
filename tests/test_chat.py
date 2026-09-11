import io
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


from app.models import Conversation, Message, MessageRole
