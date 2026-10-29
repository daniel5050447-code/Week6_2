"""
=============================================================================
[ChromaDB 안내 및 설치/사용법]
1. ChromaDB는 로컬 임베딩 벡터 데이터베이스 엔진입니다.
2. Cloud Run 환경 특징:
   - 인스턴스는 스테이트리스(Stateless) 컨테이너이며, 로컬 파일시스템(/tmp)은 임시 메모리 기반입니다.
   - 컨테이너 재시작, 콜드 스타트, 다중 인스턴스 스케일아웃 시 /tmp/chroma 내 데이터가 인스턴스 간에 동기화되지 않습니다.
   - 따라서 단일 사용자용 PoC/내부 도구 용도 배포 시에는 `--max-instances 1` 옵션을 권장합니다.
3. 로컬 테스트 실행 방법:
   - pip install -r requirements.txt
   - export GOOGLE_API_KEY="your-gemini-api-key"
   - uvicorn main:app --host 0.0.0.0 --port 8080 --reload
=============================================================================
"""

import os
import io
import json
import asyncio
import shutil
import re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

# LangChain 관련 컴포넌트 임포트
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langsmith import Client

# 환경 변수 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # 실행 시점에 환경변수 경고 로그
    print("[경고] GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

# text-embedding-004 종료 대비(대체: gemini-embedding-001)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
CHROMA_PERSIST_DIR = "/tmp/chroma"

app = FastAPI(title="Cloud Run RAG Server", version="1.0.0")

# CORS 전체 허용 (Apps Script 및 브라우저 직접 fetch 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 임베딩 및 벡터 저장소 싱글톤 헬퍼
# ---------------------------------------------------------------------------
def get_embeddings():
    """Google Gemini 임베딩 모델 인스턴스를 반환합니다."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def get_vector_store():
    """Chroma 벡터 데이터베이스 인스턴스를 반환합니다."""
    return Chroma(
        collection_name="rag_collection",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR
    )

# ---------------------------------------------------------------------------
# 실시간 스트리밍 핸들러 (BaseCallbackHandler)
# ---------------------------------------------------------------------------
class QueueStreamHandler(BaseCallbackHandler):
    """
    LLM의 새 토큰 생성 이벤트를 비동기 큐에 전달하는 핸들러.
    스레드 안전하게 asyncio.Queue로 전송합니다.
    """
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, token)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, error)

# ---------------------------------------------------------------------------
# Pydantic 요청 스키마
# ---------------------------------------------------------------------------
class BuildPage(BaseModel):
    page: int
    text: str

class BuildRequest(BaseModel):
    filename: str
    pages: List[BuildPage]

class ChatMessage(BaseModel):
    role: str # "user" 또는 "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []

# ---------------------------------------------------------------------------
# 엔드포인트 1: GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """서버 상태, 활성화 모델명, 저장된 벡터 청크 수를 반환합니다."""
    chunk_count = 0
    try:
        if os.path.exists(CHROMA_PERSIST_DIR):
            store = get_vector_store()
            chunk_count = store._collection.count()
    except Exception as e:
        chunk_count = 0

    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_chunks": chunk_count
    }

# ---------------------------------------------------------------------------
# 엔드포인트 2: POST /upload
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 전체 텍스트와 페이지별 텍스트를 추출합니다."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 형식의 파일만 지원합니다.")

    try:
        content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(content))
        
        pages_data = []
        full_text_list = []

        for idx, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            pages_data.append({
                "page": idx + 1,
                "text": page_text
            })
            full_text_list.append(f"--- Page {idx + 1} ---\n" + page_text)

        return {
            "filename": file.filename,
            "total_pages": len(pdf_reader.pages),
            "full_text": "\n\n".join(full_text_list),
            "pages": pages_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 파싱 실패: {str(e)}")

# ---------------------------------------------------------------------------
# 엔드포인트 3: POST /build (SSE 스트리밍)
# ---------------------------------------------------------------------------
@app.post("/build")
async def build_vector_store(payload: BuildRequest):
    """
    페이지별 텍스트를 분할하고 Chroma에 임베딩하여 저장합니다.
    진행 현황을 Server-Sent Events(SSE)로 전송합니다.
    """
    async def sse_event_generator():
        try:
            yield f"data: {json.dumps({'status': 'init', 'message': '기존 벡터 데이터 초기화 중...'}, ensure_ascii=False)}\n\n"
            
            # 기존 디렉토리 초기화
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            yield f"data: {json.dumps({'status': 'split', 'message': '문서 청크 분할 중...'}, ensure_ascii=False)}\n\n"
            
            documents: List[Document] = []
            for p in payload.pages:
                chunks = text_splitter.split_text(p.text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": payload.filename, "page": p.page}
                    ))

            total_chunks = len(documents)
            yield f"data: {json.dumps({'status': 'split_done', 'total_chunks': total_chunks, 'message': f'총 {total_chunks}개 청크 분할 완료'}, ensure_ascii=False)}\n\n"

            if total_chunks == 0:
                yield f"data: {json.dumps({'status': 'error', 'message': '저장할 유효한 텍스트 청크가 없습니다.'}, ensure_ascii=False)}\n\n"
                return

            # Chroma 인스턴스 생성
            store = get_vector_store()
            batch_size = 5  # API 할당량 및 안정적인 진행률 표시를 위해 5개 단위 배치

            for i in range(0, total_chunks, batch_size):
                batch = documents[i:i + batch_size]
                success = False
                retries = 0

                # Rate Limit(429) 대비 재시도 루프
                while not success and retries < 5:
                    try:
                        store.add_documents(batch)
                        success = True
                    except Exception as ex:
                        error_str = str(ex)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            retries += 1
                            wait_time = 5 * retries
                            yield f"data: {json.dumps({'status': 'retry', 'message': f'API 호출 한도 도달(429). {wait_time}초 대기 후 재시도합니다...'}, ensure_ascii=False)}\n\n"
                            await asyncio.sleep(wait_time)
                        else:
                            raise ex

                current_progress = min(i + batch_size, total_chunks)
                yield f"data: {json.dumps({'status': 'embedding', 'current': current_progress, 'total': total_chunks, 'message': f'임베딩 중 ({current_progress}/{total_chunks})'}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'status': 'done', 'total_chunks': total_chunks, 'message': '벡터 데이터베이스 생성 완료!'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# 엔드포인트 4: POST /query (SSE 스트리밍)
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_rag(payload: QueryRequest):
    """
    RAG 파이프라인 수행 후 SSE 스트리밍으로 결과를 전송합니다.
    1. 대화 기록을 고려한 질의 확장 (Augmented Query)
    2. 다중 관점 질문 생성 및 Chroma 검색 중복 제거 (MultiQuery)
    3. 현진건 작가 페르소나 및 대화기록 반영 답변 토큰 실시간 스트리밍
    """
    async def sse_query_generator():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        try:
            # 1. 질의 확장 (대화 맥락 반영)
            expansion_prompt = ChatPromptTemplate.from_messages(
