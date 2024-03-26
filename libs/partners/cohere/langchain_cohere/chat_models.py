from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import cohere
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


def get_cohere_chat_request(
    messages: List[BaseMessage],
    *,
    connectors: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Get the request for the Cohere chat API.

    Args:
        messages: The messages.
        connectors: The connectors.
        **kwargs: The keyword arguments.

    Returns:
        The request for the Cohere chat API.
    """
    documents = (
        None
        if "source_documents" not in kwargs
        else [
            {
                "snippet": doc.page_content,
                "id": doc.metadata.get("id") or f"doc-{str(i)}",
            }
            for i, doc in enumerate(kwargs["source_documents"])
        ]
    )
    kwargs.pop("source_documents", None)
    maybe_connectors = connectors if documents is None else None

    # by enabling automatic prompt truncation, the probability of request failure is
    # reduced with minimal impact on response quality
    prompt_truncation = (
        "AUTO" if documents is not None or connectors is not None else None
    )

    req = {
        "message": messages[-1].content,
        "chat_history": [
            {"role": _get_role(x), "message": x.content} for x in messages[:-1]
        ],
        "documents": documents,
        "connectors": maybe_connectors,
        "prompt_truncation": prompt_truncation,
        **kwargs,
    }

    return {k: v for k, v in req.items() if v is not None}


class ChatCohere(BaseChatModel, Serializable):
    """`Cohere` chat large language models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_cohere import ChatCohere
            from langchain_core.messages import HumanMessage

            chat = ChatCohere(cohere_api_key="my-api-key")

            messages = [HumanMessage(content="knock knock")]
            chat.invoke(messages)
    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:

    model: Optional[str] = Field(default=None)
    """Model name to use."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    cohere_api_key: Optional[SecretStr] = None
    """Cohere API key. If not provided, will be read from the environment variable."""

    stop: Optional[List[str]] = None

    streaming: bool = Field(default=False)
    """Whether to stream the results."""

    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the Cohere API key exists in the environment and instantiates the API clients."""
        values["cohere_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "cohere_api_key", "COHERE_API_KEY")
        )
        client_name = values["user_agent"]
        values["client"] = cohere.Client(
            api_key=values["cohere_api_key"].get_secret_value(),
            client_name=client_name,
        )
        values["async_client"] = cohere.AsyncClient(
            api_key=values["cohere_api_key"].get_secret_value(),
            client_name=client_name,
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cohere-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        base_params = {
            "model": self.model,
            "temperature": self.temperature,
            "stop_sequences": self.stop,
        }
        return {k: v for k, v in base_params.items() if v is not None}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)

        stream = self.client.chat_stream(**request)
        for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)

        stream = self.async_client.chat_stream(**request)
        async for data in stream:
            if data.event_type == "text-generation":
                delta = data.text
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk

    def _get_generation_info(self, response: Any) -> Dict[str, Any]:
        """Get the generation info from cohere API response."""
        return {
            "documents": response.documents,
            "citations": response.citations,
            "search_results": response.search_results,
            "search_queries": response.search_queries,
            "token_count": response.token_count,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request)

        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, "documents"):
            generation_info = self._get_generation_info(response)
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        request = get_cohere_chat_request(messages, **self._default_params, **kwargs)
        response = self.client.chat(**request)

        message = AIMessage(content=response.text)
        generation_info = None
        if hasattr(response, "documents"):
            generation_info = self._get_generation_info(response)
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        return len(self.client.tokenize(text).tokens)


def _get_role(message: BaseMessage) -> str:
    """
    Get the Cohere API representation of a role.

    Args:
        message: The message.

    Returns:
        The role of the message.

    Raises:
        ValueError: If the message is of an unknown type.
    """
    if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
        return "USER"
    elif isinstance(message, AIMessage):
        return "CHATBOT"
    elif isinstance(message, SystemMessage):
        return "SYSTEM"
    else:
        raise ValueError(f"Got unknown type {message}")
