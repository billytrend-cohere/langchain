"""Test chat model integration."""
import typing

import pytest

from langchain_cohere.chat_models import ChatCohere
from langchain_core.pydantic_v1 import SecretStr


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCohere(cohere_api_key="test")


def test_cohere_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cohere api key is a secret key."""
    # test initialization from init
    assert isinstance(ChatCohere(cohere_api_key="1").cohere_api_key, SecretStr)

    # test initialization from env variable
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    assert isinstance(ChatCohere().cohere_api_key, SecretStr)


@pytest.mark.parametrize(
    "chat_cohere,expected",
    [
        pytest.param(ChatCohere(cohere_api_key="test"), {}, id="defaults"),
        pytest.param(
            ChatCohere(cohere_api_key="test", model="foo", temperature=1.0, stop=["bar"]),
            {
                "model": "foo",
                "temperature": 1.0,
                "stop_sequences": ["bar"],
            },
            id="values are set",
        ),
    ],
)
def test_default_params(chat_cohere: ChatCohere, expected: typing.Dict) -> None:
    actual = chat_cohere._default_params
    assert expected == actual
