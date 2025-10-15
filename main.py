from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
from dotenv import load_dotenv
from agent import generate_initial_code, update_existing_code

load_dotenv()

app = FastAPI(title="Garliq AI Agent Service - Next Gen")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================

class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama-3.3-70b"
    current_code: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []
    user_id: str = None
    session_id: str = None

class GenerateResponse(BaseModel):
    html: str
    success: bool
    total_tokens: int = 0
    token_usage: Dict[str, Any] = {}
    error: Optional[str] = None


# ==================== HEALTH CHECK ====================

@app.get("/")
def read_root():
    return {
        "status": "Garliq AI Agent Service - Enhanced Edition",
        "version": "2.0.0",
        "features": [
            "Real-time web search via Perplexity AI",
            "Actual token usage tracking",
            "Daily generation limits for free tier",
            "Enhanced micro-app generation",
            "Automatic citations and sources"
        ]
    }

@app.get("/health")
def health_check():
    perplexity_configured = bool(os.getenv("PERPLEXITY_API_KEY"))
    return {
        "status": "healthy",
        "perplexity_configured": perplexity_configured,
        "features": {
            "search": "Perplexity AI unified (research + web)" if perplexity_configured else "Not configured",
            "real_time_data": perplexity_configured,
            "citations": perplexity_configured,
            "token_tracking": True
        }
    }


# ==================== STREAMING GENERATION ====================

async def generate_stream(request: GenerateRequest):
    """Stream generation progress to frontend with token tracking"""
    tokens_used = 0
    generation_success = False
    html_result = ""
    
    try:
        # Validate model
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid model selected'})}\n\n"
            return
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing AI agent...'})}\n\n"
        await asyncio.sleep(0.3)
        
        # Tool availability message for Claude
        if request.model == "claude-sonnet-4.5":
            perplexity_status = "🌐 Web search tools activated (Perplexity AI)..." if os.getenv("PERPLEXITY_API_KEY") else "⚠️ Web search not configured"
            yield f"data: {json.dumps({'type': 'status', 'message': perplexity_status})}\n\n"
            await asyncio.sleep(0.4)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your prompt...'})}\n\n"
        await asyncio.sleep(0.5)
        
        if request.current_code:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Reading current code structure...'})}\n\n"
            await asyncio.sleep(0.4)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Planning modifications...'})}\n\n"
            await asyncio.sleep(0.3)
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Designing micro-app architecture...'})}\n\n"
            await asyncio.sleep(0.4)
        
        # Research phase for Claude
        if request.model == "claude-sonnet-4.5" and os.getenv("PERPLEXITY_API_KEY"):
            yield f"data: {json.dumps({'type': 'status', 'message': '🔍 Researching real-time data sources...'})}\n\n"
            await asyncio.sleep(0.8)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Writing HTML structure...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Crafting modern CSS styles...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Adding JavaScript interactions...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': '⚡ Generating code with AI...'})}\n\n"
        
        # Generate code with token tracking
        if request.current_code:
            result = update_existing_code(
                current_code=request.current_code,
                user_message=request.prompt,
                chat_history=[msg.dict() for msg in request.chat_history],
                model_name=request.model
            )
        else:
            result = generate_initial_code(
                prompt=request.prompt,
                model_name=request.model
            )
        
        html_result = result['html']
        tokens_used = result['total_tokens']
        token_usage = result['token_usage']
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Finalizing micro-app...'})}\n\n"
        await asyncio.sleep(0.3)
        
        generation_success = True
        
        # Send complete result with token info
        yield f"data: {json.dumps({'type': 'complete', 'html': html_result, 'total_tokens': tokens_used, 'token_usage': token_usage})}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Generation error: {error_msg}")
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'tokens_used': tokens_used})}\n\n"


@app.post("/generate")
async def generate_code_stream(request: GenerateRequest):
    """Streaming endpoint for code generation"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==================== FALLBACK SYNC ENDPOINT ====================

@app.post("/generate-sync", response_model=GenerateResponse)
async def generate_code_sync(request: GenerateRequest):
    """Non-streaming endpoint for compatibility"""
    try:
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            raise HTTPException(status_code=400, detail="Invalid model selected")
        
        if request.current_code:
            result = update_existing_code(
                current_code=request.current_code,
                user_message=request.prompt,
                chat_history=[msg.dict() for msg in request.chat_history],
                model_name=request.model
            )
        else:
            result = generate_initial_code(
                prompt=request.prompt,
                model_name=request.model
            )
        
        return GenerateResponse(
            html=result['html'],
            success=True,
            total_tokens=result['total_tokens'],
            token_usage=result['token_usage']
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return GenerateResponse(
            html="",
            success=False,
            error=str(e),
            total_tokens=0
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Garliq AI Agent Service on port {port}")
    print(f"🔑 Perplexity API: {'✅ Configured' if os.getenv('PERPLEXITY_API_KEY') else '❌ Not configured'}")
    print(f"🌐 Features: Real-time web search, Citations, Token tracking")
    uvicorn.run(app, host="0.0.0.0", port=port)