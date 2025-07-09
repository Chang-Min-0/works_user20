from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
#from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import asyncio
from datetime import datetime
import uvicorn

# 로컬 모듈 import
from agents.openAi_agent import get_chatbot_manager


# FastAPI 앱 생성
app = FastAPI(
    title="OpenAI 채팅봇 API",
    description="FastAPI와 OpenAI를 활용한 AI 채팅봇",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 경로
    redoc_url="/redoc"  # ReDoc 경로
)

# 정적 파일 및 템플릿 설정
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic 모델 정의
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str  # 사용자 메시지 (필수)
    reset_conversation: Optional[bool] = False  # 대화 기록 초기화 여부

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    success: bool  # 처리 성공 여부
    response: str  # AI 응답 메시지
    timestamp: str  # 응답 생성 시간
    processing_time: float  # 처리 시간 (초)
    agent_info: Dict[str, Any]  # Agent 정보

class HealthResponse(BaseModel):
    """서버 상태 응답 모델"""
    status: str
    message: str
    timestamp: str
    agent_status: Dict[str, Any]

# 전역 변수
startup_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업"""
    print("🚀 FastAPI 채팅봇 서버 시작")
    print(f"⏰ 시작 시간: {startup_time}")
    
    try:
        # Agent 관리자 미리 초기화
        manager = get_chatbot_manager()
        print("✅ Agent 관리자 초기화 완료")
    except Exception as e:
        print(f"❌ 서버 시작 중 오류: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    print("🛑FastAPI 채팅봇 서버 종료")

# API 엔드포인트

@app.get("/main", response_class=HTMLResponse)
async def root(request: Request):
    """메인 페이지 - 채팅 인터페이스"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인 API"""
    try:
        # Agent 관리자 상태 확인
        manager = get_chatbot_manager()
        agent_info = manager.get_agent_info()
        
        return HealthResponse(
            status="healthy",
            message="서버가 정상적으로 실행 중입니다.",
            timestamp=datetime.now().isoformat(),
            agent_status=agent_info
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"서버 상태 확인 중 오류: {str(e)}",
            timestamp=datetime.now().isoformat(),
            agent_status={}
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    채팅 API 엔드포인트
    
    사용자 메시지를 받아 OpenAI API를 통해 AI 응답을 생성합니다.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # 입력 검증
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400, 
                detail="메시지가 비어있습니다. 내용을 입력해주세요."
            )
        
        # Agent 관리자 가져오기
        manager = get_chatbot_manager()
        
        # 대화 기록 초기화 요청 처리
        if request.reset_conversation:
            manager.reset_conversation()
            print("🔄 사용자 요청으로 대화 기록 초기화")
        
        # AI 응답 생성
        print(f"📝 사용자 메시지: {request.message}")
        ai_response = await manager.get_response_with_search(request.message)
        
        # 처리 시간 계산
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 응답 생성
        response = ChatResponse(
            success=True,
            response=ai_response,
            timestamp=datetime.now().isoformat(),
            processing_time=round(processing_time, 2),
            agent_info=manager.get_agent_info()
        )
        
        print(f"✅ 응답 생성 완료 (처리시간: {processing_time:.2f}초)")
        return response
        
    except HTTPException:
        # HTTP 예외는 그대로 재발생
        raise
    except Exception as e:
        # 기타 예외 처리
        error_msg = f"채팅 처리 중 오류가 발생했습니다: {str(e)}"
        print(f"❌ {error_msg}")
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ChatResponse(
            success=False,
            response=error_msg,
            timestamp=datetime.now().isoformat(),
            processing_time=round(processing_time, 2),
            agent_info={}
        )

@app.post("/reset")
async def reset_conversation():
    """대화 기록 초기화 API"""
    try:
        manager = get_chatbot_manager()
        manager.reset_conversation()
        
        return JSONResponse(
            content={
                "success": True,
                "message": "대화 기록이 초기화되었습니다.",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        error_msg = f"대화 기록 초기화 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/info")
async def get_agent_info():
    """Agent 정보 조회 API"""
    try:
        manager = get_chatbot_manager()
        agent_info = manager.get_agent_info()
        
        return JSONResponse(
            content={
                "success": True,
                "data": agent_info,
                "server_uptime": str(datetime.now() - startup_time),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        error_msg = f"Agent 정보 조회 중 오류: {str(e)}"
        print(f"❌ {error_msg}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )

# 서버 실행 함수
def run_server():
    """서버 실행"""
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "True").lower() == "true",
        log_level="info"
    )

if __name__ == "__main__":
    run_server()