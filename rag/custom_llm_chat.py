# custom_litellm_chat.py

import copy
import json
import logging
from typing import List, Dict, Optional

import litellm  # Ensure litellm is installed: pip install litellm
import pathway as pw
from pathway.udfs import AsyncRetryStrategy, CacheStrategy
from pathway.xpacks.llm.llms import BaseChat
from pathway.xpacks.llm._utils import _check_model_accepts_arg  # Ensure this utility is correctly imported

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLiteLLMChat(BaseChat):
    """
    Custom Pathway wrapper for LiteLLM Chat services using litellm.

    Inherits from BaseChat and implements the __wrapped__ method for integration with Pathway.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        retry_strategy: Optional[AsyncRetryStrategy] = None,
        cache_strategy: Optional[CacheStrategy] = None,
        model: Optional[str] = None,
        **litellm_kwargs,
    ):
        super().__init__(
            executor=pw.udfs.async_executor(capacity=capacity, retry_strategy=retry_strategy),
            cache_strategy=cache_strategy,
        )
        self.litellm_kwargs = litellm_kwargs.copy()
        if model is not None:
            print(f"Setting model to {model}")
            self.litellm_kwargs["model"] = model

    def _check_model_accepts_arg(self, model: str, provider: str, arg_name: str) -> bool:
        """
        Wrapper to the standalone _check_model_accepts_arg function.
        """
        return _check_model_accepts_arg(model, provider, arg_name)

    def _accepts_call_arg(self, arg_name: str) -> bool:
        """
        Check whether the LLM accepts the argument during the inference.
        If `model` is not set, return `False`.
        """
        model = self.kwargs.get("model")
        if model is None:
            return False

        # Assuming the provider is 'litellm' for LiteLLM models
        provider = "litellm"  # Adjust if using different providers
        return self._check_model_accepts_arg(model, provider, arg_name)
    def __wrapped__(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Synchronously sends messages to LiteLLM and returns the response.

        Args:
            messages (List[Dict[str, str]]): List of messages with 'role' and 'content'.
            **kwargs: Override for litellm_kwargs.

        Returns:
            Optional[str]: LLM response or None.
        """
        logger.info("Entering __wrapped__ method.")

        # Merge default and overridden kwargs
        request_kwargs = {**self.litellm_kwargs, **kwargs}
        logger.debug(f"Request kwargs: {request_kwargs}")

        # Log the request event
        event = {
            "_type": "lite_llm_chat_request",
            "kwargs": copy.deepcopy(request_kwargs),
            "messages": messages,
        }
        logger.info(json.dumps(event))

        try:
            # Call litellm completion synchronously
            response = litellm.completion(messages=messages, **request_kwargs)
            logger.info("Received response from litellm.completion.")

            # Extract the content from the response
            llm_response = response.choices[0]["message"]["content"].strip()
            logger.debug(f"Extracted LLM response: {llm_response}")

            # Log the response event
            event = {
                "_type": "lite_llm_chat_response",
                "response": llm_response,
            }
            logger.info(json.dumps(event))
            return llm_response
        except Exception as e:
            logger.error(f"Error generating response from LiteLLM: {e}")
            return None

    def __call__(self, messages: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """
        Integrates with Pathway's UDF system.

        Args:
            messages (pw.ColumnExpression): Column containing message lists.
            **kwargs: Override for litellm_kwargs.

        Returns:
            pw.ColumnExpression: Column with LLM responses.
        """
        print("Inside __call__")
        return super().__call__(messages, **kwargs)
    
    def get_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Helper method to directly call the __wrapped__ method.
        """
        return self.__wrapped__(messages, **kwargs)