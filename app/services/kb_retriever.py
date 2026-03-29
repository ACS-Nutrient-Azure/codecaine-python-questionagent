"""
kb_retriever.py

Analysis Agent와 동일한 KB 사용 (numpy + Cohere Bedrock)

KB 파일:
  kb.npz        — 정규화된 임베딩 벡터 (252, 1536) float32
  kb_texts.json — 문서 텍스트 + 메타데이터

벡터 검색: numpy cosine similarity (브루트포스)
임베딩 모델: cohere.embed-multilingual-v3 (AWS Bedrock, 1024차원)
"""
import json
import logging

import boto3
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

_COHERE_MODEL_ID = "cohere.embed-multilingual-v3"


def _embed_query(text: str) -> np.ndarray:
    """Cohere Bedrock으로 쿼리 임베딩 후 정규화된 벡터 반환."""
    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    response = client.invoke_model(
        modelId=_COHERE_MODEL_ID,
        body=json.dumps({
            "texts": [text],
            "input_type": "search_query",
            "embedding_types": ["float"],
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    vec = np.array(result["embeddings"]["float"][0], dtype=np.float32)
    return vec / np.linalg.norm(vec)


# 컨테이너 내 캐싱
_vectors = None
_texts = None


def _get_kb():
    global _vectors, _texts
    if _vectors is not None:
        return _vectors, _texts

    vectors_path = f"{settings.KB_LOCAL_PATH}/kb.npz"
    texts_path = f"{settings.KB_LOCAL_PATH}/kb_texts.json"

    _vectors = np.load(vectors_path)["vectors"]
    with open(texts_path, encoding="utf-8") as f:
        _texts = json.load(f)

    logger.info(f"[KB] 로드 완료 — {_vectors.shape[0]}개 청크, {_vectors.shape[1]}차원")
    return _vectors, _texts


def retrieve(query: str) -> str:
    """
    쿼리와 의미적으로 유사한 청크를 KB_TOP_K개 검색해 하나의 문자열로 반환.
    검색 실패 시 빈 문자열 반환 (에이전트가 다른 tool로 fallback 가능하도록).
    """
    try:
        vectors, texts = _get_kb()
        query_vec = _embed_query(query)

        # cosine similarity
        similarities = vectors @ query_vec
        top_indices = np.argsort(similarities)[::-1][:settings.KB_TOP_K]

        docs = [texts["documents"][i] for i in top_indices]
        if not docs:
            logger.info(f"[KB] 검색 결과 없음: {query}")
            return ""

        context = "\n\n".join(docs)
        logger.info(f"[KB] {len(docs)}개 청크 검색됨 (query: {query[:50]})")
        return context

    except Exception as e:
        logger.warning(f"[KB] 검색 실패: {e}")
        return ""
