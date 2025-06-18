from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Optional

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding

OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
OPENAI_EMBED_BATCH_SIZE = 128


@dataclass
class FilterOptions:
    filter_model_name: str
    filter_count: int


def get_embedding_model(model_name: str) -> BaseEmbedding:
    if model_name in OPENAI_EMBEDDING_MODELS:
        return OpenAIEmbedding(
            model_name=model_name, embed_batch_size=OPENAI_EMBED_BATCH_SIZE
        )

    raise Exception("Unsupported embedding model")


def is_non_empty_directory(path: Path) -> bool:
    if path.exists() and path.is_dir():
        for _ in path.iterdir():
            return True

    return False


def get_file_path_to_contents(
    file_paths: list[Path], logger: Logger
) -> dict[Path, str]:
    file_path_to_contents = {}
    for file_path in file_paths:
        try:
            with file_path.open() as file:
                file_path_to_contents[file_path] = file.read()
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e, exc_info=e)

    return file_path_to_contents


def create_new_index(
    index_dir: Path,
    file_path_to_contents: dict[Path, str],
    embedding_model: BaseEmbedding,
    logger: Logger,
) -> VectorStoreIndex:
    logger.info("Creating new index at %s", index_dir)
    Settings.embed_model = embedding_model

    pass


def load_index(
    index_dir: Path, embedding_model: BaseEmbedding, logger: Logger
) -> Optional[VectorStoreIndex]:
    logger.info(
        "Attempting to load index from storage at %s",
        index_dir,
    )
    Settings.embed_model = embedding_model

    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context)
    except Exception as e:
        logger.error(
            "Failed to load index from storage at %s: %s",
            index_dir,
            e,
            exc_info=e,
        )
        return None


def load_or_create_index(
    index_dir: Path,
    file_path_to_contents: dict[Path, str],
    embedding_model: BaseEmbedding,
    logger: Logger,
) -> VectorStoreIndex:
    index = None
    if is_non_empty_directory(index_dir):
        index = load_index(index_dir, embedding_model, logger)
    if index is None:
        index = create_new_index(
            index_dir, file_path_to_contents, embedding_model, logger
        )
    return index


def do_filter_retrieval(
    filter_options: FilterOptions,
    filter_index_dir: Path,
    file_paths: list[Path],
    prompt: str,
    logger: Logger,
) -> dict[Path, str]:
    logger.info("Doing filter retrieval with options: %s", filter_options)
    filter_model = get_embedding_model(filter_options.filter_model_name)
    file_path_to_contents = get_file_path_to_contents(file_paths, logger)
    filter_index = load_or_create_index(
        filter_index_dir, file_path_to_contents, filter_model, logger
    )
    filter_retriever = VectorIndexRetriever(
        index=filter_index,
        embed_model=filter_model,
        similarity_top_k=filter_options.filter_count,
    )
    filtered_documents = filter_retriever.retrieve(prompt)
    logger.info("Retrieved %s sections")

    filtered_file_path_to_sections: dict[Path, list[str]] = {}
    for node in filtered_documents:
        file_name = node.metadata["File Name"]
        sections = filtered_file_path_to_sections.setdefault(file_name, [])
        sections.append(node.text)

    filtered_file_path_to_contents = {
        file_path: "\n...\n".join(file_sections)
        for file_path, file_sections in filtered_file_path_to_sections.items()
    }
    return filtered_file_path_to_contents


def load_or_create_retrieval_index(
    embeddings_dir: Path,
    prompt: str,
    file_paths: list[Path],
    filter_options: Optional[FilterOptions],
    embedding_model_name: str,
    logger: Logger,
) -> VectorStoreIndex:
    embedding_model = get_embedding_model(embedding_model_name)
    retrieval_index_dir = embeddings_dir / "retrieval_index"
    retrieval_index_exists = is_non_empty_directory(retrieval_index_dir)
    retrieval_index: Optional[VectorStoreIndex] = None

    if retrieval_index_exists:
        logger.info(
            "Retrieval index exists at %s, trying to load directly",
            retrieval_index_dir,
        )
        Settings.embed_model = embedding_model
        retrieval_index = load_index(retrieval_index_dir, embedding_model, logger)

    if retrieval_index is None:
        logger.info("Retrieval index has not been loaded, will create it")
        if filter_options:
            filter_index_dir = embeddings_dir / "filter_index"
            file_path_to_contents = do_filter_retrieval(
                filter_options, filter_index_dir, file_paths, prompt, logger
            )
        else:
            file_path_to_contents = get_file_path_to_contents(file_paths, logger)

        retrieval_index = create_new_index(
            retrieval_index_dir, file_path_to_contents, embedding_model, logger
        )

    return retrieval_index


def retrieve(
    embeddings_dir: Path,
    prompt: str,
    file_paths: list[Path],
    filter_options: Optional[FilterOptions],
    embedding_model_name: str,
    just_create_retrieval_index: bool,
    retrieve_count: int,
    logger: Logger,
) -> list[tuple[Path, str]]:
    retrieval_index = load_or_create_retrieval_index(
        embeddings_dir, prompt, file_paths, filter_options, embedding_model_name, logger
    )
    if just_create_retrieval_index:
        return []

    retriever = VectorIndexRetriever(
        index=retrieval_index,
    )
    retrieved_documents = retriever.retrieve(prompt)
    logger.info("Retrieved %s sections using ")
