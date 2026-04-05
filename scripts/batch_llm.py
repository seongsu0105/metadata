from __future__ import annotations

import time

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout, Timeout

from batch_config import (
    METADATA_SYSTEM,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_HTTP_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_READ_TIMEOUT,
    OLLAMA_URL,
    effective_metadata_max_chars,
    effective_ollama_num_predict,
    metadata_user_prompt,
)


def dataset_user_prompt(text: str, bc_block: str) -> str:
    """Ollama 호출·finetune_dataset input 공통 — Worker metadata_user_prompt 와 동일."""
    return metadata_user_prompt(
        text, bc_block, max_summary_chars=effective_metadata_max_chars()
    )


def generate_summary_ollama(text: str, bc_block: str) -> str:
    prompt = dataset_user_prompt(text, bc_block)
    payload = {
        "model": OLLAMA_MODEL,
        "system": METADATA_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": effective_ollama_num_predict(),
        },
    }
    timeout = (OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT)
    last_err: BaseException | None = None
    for attempt in range(OLLAMA_HTTP_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except (ReadTimeout, Timeout) as e:
            last_err = e
            if attempt < OLLAMA_HTTP_RETRIES:
                wait = 5 * (attempt + 1)
                print(
                    f"  Ollama 응답 지연(타임아웃), {wait}초 후 재시도 "
                    f"({attempt + 1}/{OLLAMA_HTTP_RETRIES})…"
                )
                time.sleep(wait)
        except RequestsConnectionError as e:
            last_err = e
            if attempt < OLLAMA_HTTP_RETRIES:
                wait = 5 * (attempt + 1)
                print(
                    f"  Ollama 연결 실패, {wait}초 후 재시도 "
                    f"({attempt + 1}/{OLLAMA_HTTP_RETRIES})…"
                )
                time.sleep(wait)
    assert last_err is not None
    raise last_err
