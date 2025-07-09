import os
import uuid
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# AZURESEARCH_FIELDS_CONTENT: 자신의 인덱스에서 검색 대상을 의미하는 필드명
# AZURESEARCH_FIELDS_CONTENT_VECTOR: 'AZURESEARCH_FIELDS_CONTENT' 의 임베딩을 의미하는 필드명
os.environ['AZURESEARCH_FIELDS_CONTENT'] = "chunk"
os.environ['AZURESEARCH_FIELDS_CONTENT_VECTOR'] = "text_vector"

from openai import AzureOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정
class AzureConfig:
    API_TYPE = os.getenv("AZURE_API_TYPE", "azure")
    API_BASE = os.getenv("AZURE_ENDPOINT")
    API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    API_KEY = os.getenv("AZURE_API_KEY")
    MODEL = os.getenv("AZURE_MODEL")
    TEMPERATURE = 0.8
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYENT_NAME")

# Azure AI Search 설정
class AzureSearchConfig:
    SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
    KEY = os.getenv("AZURE_SEARCH_KEY")
    INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    ENDPOINT = f"https://{SERVICE}.search.windows.net" if SERVICE else None
    
# OpenAI 클라이언트 초기화
client = AzureOpenAI(
    api_key=AzureConfig.API_KEY,
    api_version=AzureConfig.API_VERSION,
    azure_endpoint=AzureConfig.API_BASE
)

class VectorStoreManager:
    """벡터 스토어 관리 클래스"""
    
    def __init__(self):
        self._initialize_embeddings()
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """임베딩 초기화"""
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AzureConfig.EMBEDDING_DEPLOYMENT,
            openai_api_version=AzureConfig.API_VERSION,
            azure_endpoint=AzureConfig.API_BASE,
            api_key=AzureConfig.API_KEY,
        )
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        self.vector_store = AzureSearch(
            azure_search_endpoint=AzureSearchConfig.AZURE_SEARCH_ENDPOINT,
            azure_search_key=AzureSearchConfig.KEY,
            index_name=AzureSearchConfig.INDEX_NAME,
            embedding_function=self.embeddings.embed_query,
            search_type="hybrid",
            additional_search_client_options={
                "retry_total": 3, 
                "api_version": "2025-05-01-preview"
            },
        )
    
    def get_retriever(self, top_k=3):
        """검색 리트리버 반환"""
        return self.vector_store.as_retriever(
            search_type="hybrid",
            k=top_k
        )


class SimpleChatbotManager:
    """채팅봇 관리 클래스"""

    def __init__(self):
        self.model = AzureConfig.MODEL
        self.temperature = AzureConfig.TEMPERATURE
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.system_prompt = (
            "당신은 Azure OpenAI와 Azure AI Search를 활용한 RAG(Retrieval-Augmented Generation)기반 AI 어시스턴트입니다. "
            "사용자 질문에 대해 자체 지식이 아닌 RAG 기반의 정보만을 바탕으로 답변해야 합니다. "
            "항상 검색된 내용 안에서 최대한 정확하고 근거 있는 정보를 요약해 전달하고, "
            "답변에 활용한 출처를 명확히 밝혀주세요. "
            "검색 결과가 부족하거나 관련 정보가 없다면, 이를 솔직하게 고지하고 추측하지 마세요.\n"
            "응답은 다음 형식을 따라주세요:\n"
            "1. 주요 정보를 명확하게 구조화된 단락으로 제시\n"
            "2. 중요한 개념이나 용어는 마크다운 **굵은 글씨**로 강조\n"
            "3. 필요한 경우 내용을 목록화하여 가독성 향상\n"
            "4. 응답 마지막에 '출처: [문서명]' 형식으로 RAG참고 출처 명시"
        )
        # OpenAI 클라이언트 초기화
        self.client = AzureOpenAI(
            api_key=AzureConfig.API_KEY,
            api_version=AzureConfig.API_VERSION,
            azure_endpoint=AzureConfig.API_BASE
        )
        
        # 벡터 스토어 초기화
        self.vector_store_manager = self._init_vector_store()
        
    def _init_vector_store(self):
        """벡터 스토어 초기화"""
        try:
            if all([AzureSearchConfig.SERVICE, AzureSearchConfig.KEY, AzureSearchConfig.INDEX_NAME]):
                vector_store = VectorStoreManager()
                print("✅ Azure AI Search 벡터 스토어 초기화 완료")
                return vector_store
            else:
                print("⚠️ Azure AI Search 환경변수가 설정되어 있지 않습니다.")
                return None
        except Exception as e:
            print(f"❌ 벡터 스토어 초기화 실패: {str(e)}")
            return None
    
    async def search_documents(self, query: str, top: int = 3) -> List[Dict[str, Any]]:
        """문서 검색"""
        if not self.vector_store_manager:
            return [{"error": "Azure AI Search가 설정되지 않았습니다."}]
        
        try:
            retriever = self.vector_store_manager.get_retriever(top_k=top)
            docs = retriever.invoke(query)
            
            # Document 객체를 딕셔너리로 변환
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return results
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {str(e)}")
            return [{"error": f"검색 중 오류 발생: {str(e)}"}]
    
    def _prepare_search_context(self, search_results):
        """검색 결과를 컨텍스트로 변환"""
        if not search_results:
            return "검색 결과를 찾을 수 없습니다. 질문에 대한 답변을 생성합니다."
            
        search_context = "다음은 질문과 관련된 검색 결과입니다:\n\n"
        for i, result in enumerate(search_results):
            search_context += f"[{i+1}] {json.dumps(result, ensure_ascii=False)}\n\n"
        search_context += "위 정보를 참고하여 사용자 질문에 답변해주세요."
        
        return search_context
    
    async def _execute_search(self, user_message: str):
        """검색 실행 및 결과 처리"""
        search_results = await self.search_documents(user_message)
        
        # 오류가 없는 검색 결과인지 확인
        has_error = any(isinstance(result.get("error", None), str) for result in search_results)
        
        if search_results and not has_error:
            return self._prepare_search_context(search_results)
        else:
            print("⚠️ 검색 결과가 없거나 오류가 발생했습니다.")
            return "검색 결과를 찾을 수 없습니다. 질문에 대한 답변을 생성합니다."
    
    async def get_response_with_search(self, user_message: str, use_search: bool = True) -> str:
        """AI 응답 생성 (검색 기능 포함)"""
        try:
            print(f"🔄 메시지 처리 시작: {user_message[:50]}...")
            
            # 사용자 메시지 대화 기록에 추가
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # 메시지 준비
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 검색 기능 사용 시 관련 정보 검색
            if use_search and self.vector_store_manager:
                search_context = await self._execute_search(user_message)
                messages.append({"role": "system", "content": search_context})
        
            # 대화 기록 추가
            messages.extend(self.conversation_history)
            
            # OpenAI API 호출 (최신 방식)
            response = self._call_openai_api(messages)
            reply = response.choices[0].message.content
            
            # AI 응답을 대화 기록에 추가
            self.conversation_history.append({"role": "assistant", "content": reply})
            
            print(f"✅ 응답 생성 완료: {len(reply)}자")
            return reply

        except Exception as e:
            error_msg = f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _call_openai_api(self, messages):
        """OpenAI API 호출 (최신 버전)"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
        print("🔄 대화 기록 초기화 완료")

    def get_agent_info(self) -> Dict[str, Any]:
        """Agent 정보 반환"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "session_id": self.session_id,
            "conversation_turns": len(self.conversation_history) // 2,
            "search_enabled": self.vector_store_manager is not None
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """현재 대화 세션의 전체 기록 반환"""
        return self.conversation_history
    
    def set_system_prompt(self, prompt: str) -> None:
        """시스템 프롬프트 설정"""
        self.system_prompt = prompt
        print(f"✅ 시스템 프롬프트 설정 완료")


# 싱글톤 인스턴스
_chatbot_manager_instance = None

def get_chatbot_manager() -> SimpleChatbotManager:
    """싱글톤 패턴으로 Chatbot 관리자 반환"""
    global _chatbot_manager_instance
    if _chatbot_manager_instance is None:
        _chatbot_manager_instance = SimpleChatbotManager()
    return _chatbot_manager_instance