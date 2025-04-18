from typing import Any
from langchain_aws import BedrockEmbeddings


def get_embedding_function() -> Any:
    """
    Returns an instance of BedrockEmbeddings.

    Returns:
        Any: An instance of BedrockEmbeddings.
    """
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",  # Specified the titan model we would use
        region_name="us-east-1",
    )
