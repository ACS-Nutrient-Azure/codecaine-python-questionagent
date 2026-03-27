"""
kb_retriever.py

컨테이너 이미지에 포함된 ChromaDB에서 영양소/의약품 관련 전문 지식을 검색.

- KB 데이터: analysisagent의 lpi_vector_db를 그대로 복사한 kb_vector_db/
- 임베딩 모델: ChromaDB 기본 임베딩 (원본 DB와 동일)
- 컨테이너 내 _collection 캐싱으로 요청마다 재로드하지 않음
"""
import logging

import chromadb

from app.core.config import settings

logger = logging.getLogger(__name__)

# 모듈 레벨 캐시 — 컨테이너 재시작 전까지 유지
_collection = None


def _get_collection():
    """ChromaDB collection을 최초 1회만 로드하고 이후 캐시 반환."""
    global _collection
    if _collection is not None:
        return _collection
    
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name="text-embedding-ada-002",
    )
    client = chromadb.PersistentClient(path=settings.KB_LOCAL_PATH)
    _collection = client.get_collection(name=settings.KB_COLLECTION_NAME, embedding_function=embedding_fn)
    logger.info(f"[KB] collection 로드 완료 — {_collection.count()}개 청크")
    return _collection


def retrieve(query: str) -> str:
    """
    쿼리와 의미적으로 유사한 청크를 KB_TOP_K개 검색해 하나의 문자열로 반환.
    검색 실패 시 빈 문자열 반환 (에이전트가 다른 tool로 fallback 가능하도록).
    """
    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=settings.KB_TOP_K)
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception as e:
        logger.warning(f"[KB] 검색 실패: {e}")
        return ""
