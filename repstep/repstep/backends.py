import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import openai
from openai import OpenAI

MAX_RETRIES = 5

RETRY_LONG_WAIT_TIME_SECONDS = 5
RETRY_SHORT_WAIT_TIME_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 100

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, image_urls: list[str] = [], max_retries: int = MAX_RETRIES
    ) -> list[str]:
        pass


class OpenAIGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        api_key = os.environ.get(ENV_OPENAI_API_KEY)
        if api_key is None:
            raise Exception(f"Missing API key: {ENV_OPENAI_API_KEY}")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, image_urls=[], max_retries=MAX_RETRIES) -> list[str]:
        config = self.build_config(prompt, image_urls)

        result: Optional[list[str]] = None
        retries = 0

        while retries <= MAX_RETRIES:
            retry_wait_time: Optional[float] = None

            try:
                completion = self.client.chat.completions.create(**config)
                if completion is None or not hasattr(completion, "choices"):
                    logger.error("Invalid response received: %s", completion)
                    raise Exception("Invalid API response")

                result = [choice.message.content for choice in completion.choices]
                break
            except openai.OpenAIError as e:
                if isinstance(e, openai.BadRequestError):
                    logger.error("Invalid request: %s", e, exc_info=e)
                    raise e
                elif isinstance(e, openai.RateLimitError):
                    logger.info("Rate limit exceeded: %s", e, exc_info=e)
                    retry_wait_time = RETRY_LONG_WAIT_TIME_SECONDS
                elif isinstance(e, openai.APIConnectionError):
                    logger.info("API connection error: %s", e, exc_info=e)
                    retry_wait_time = RETRY_LONG_WAIT_TIME_SECONDS
                elif isinstance(e, openai.APITimeoutError):
                    logger.info(
                        "Request timed out after %s seconds: %s",
                        REQUEST_TIMEOUT_SECONDS,
                        e,
                        exc_info=e,
                    )
                    retry_wait_time = RETRY_SHORT_WAIT_TIME_SECONDS
                else:
                    logger.info("Unknown error: %s", e, exc_info=e)
                    retry_wait_time = RETRY_SHORT_WAIT_TIME_SECONDS

            assert retry_wait_time is not None
            logger.info("Waiting %s seconds before retrying...", retry_wait_time)
            time.sleep(retry_wait_time)

            retries += 1

        if result is None:
            raise Exception(f"Max retries ({MAX_RETRIES}) exceeded")

        return result

    def build_config(self, prompt: str, image_urls: list[str] = []) -> dict[str, Any]:
        if image_urls:
            content: list[Any] = [{"type": "text", "text": prompt}]
            for image_url in image_urls:
                image_data = {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
                content.append(image_data)
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        config = {"model": self.model_name, "temperature": 0, "messages": messages}
        return config


def get_generator_from_model_name(model_name: str) -> BaseGenerator:
    if model_name == "gpt-4o-mini":
        return OpenAIGenerator(model_name)

    raise Exception(f"Unsupported model: {model_name}")
