from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
from dotenv import load_dotenv
from agent import generate_initial_code, update_existing_code
from supabase import create_client, Client

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

class AsyncGenerateRequest(GenerateRequest):
    supabase_url: str
    supabase_key: str

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
        "version": "2.1.0",
        "features": [
            "Real-time web search via Perplexity AI",
            "Actual token usage tracking",
            "Daily generation limits for free tier",
            "Enhanced micro-app generation",
            "Automatic citations and sources",
            "Async background processing"
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
            "token_tracking": True,
            "async_processing": True
        }
    }


# ==================== ASYNC BACKGROUND GENERATION ====================

async def process_generation_background(request: AsyncGenerateRequest):
    """Process generation in background and update Supabase"""
    supabase: Client = create_client(request.supabase_url, request.supabase_key)
    
    tokens_used = 0
    generation_success = False
    html_result = ""
    
    try:
        print(f"🔥 Starting background generation for session: {request.session_id}")
        
        # Generate code
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
        generation_success = True
        
        print(f"✅ Generation complete. Tokens used: {tokens_used}")
        
        # ==================== SAVE TO DATABASE ====================
        if html_result and generation_success:
            # Save chat messages
            supabase.table('chat_messages').insert({
                'session_id': request.session_id,
                'role': 'user',
                'content': request.prompt
            }).execute()

            supabase.table('chat_messages').insert({
                'session_id': request.session_id,
                'role': 'assistant',
                'content': 'Generated code successfully'
            }).execute()

            # Update project's html_code
            supabase.table('projects').update({ 
                'html_code': html_result,
                'updated_at': 'now()'
            }).eq('session_id', request.session_id).execute()

            print('✅ Project html_code updated')

            # Update status to completed
            supabase.table('sessions').update({ 
                'generation_status': 'completed',
                'generation_error': None
            }).eq('id', request.session_id).execute()

            # ==================== DEDUCT TOKENS FOR CLAUDE ====================
            if request.model == 'claude-sonnet-4.5':
                tokens_to_deduct = tokens_used if tokens_used > 0 else 2000
                
                print(f"🔍 Deducting {tokens_to_deduct} tokens for user {request.user_id}")
                
                result = supabase.rpc('deduct_tokens_with_tracking', {
                    'p_user_id': request.user_id,
                    'p_amount': tokens_to_deduct,
                    'p_description': f'Code generation with {request.model} ({tokens_to_deduct} tokens)',
                    'p_session_id': request.session_id
                }).execute()

                if result.data:
                    print(f"✅ Tokens deducted. New balance: {result.data.get('new_balance')}")
                else:
                    print(f"⚠️ Token deduction issue: {result}")

        else:
            # Generation failed
            supabase.table('sessions').update({ 
                'generation_status': 'failed',
                'generation_error': 'No HTML generated'
            }).eq('id', request.session_id).execute()

            # FAILURE: Deduct 2000 tokens for Claude
            if request.model == 'claude-sonnet-4.5':
                supabase.rpc('deduct_tokens_with_tracking', {
                    'p_user_id': request.user_id,
                    'p_amount': 2000,
                    'p_description': f'Generation failure penalty - {request.model}',
                    'p_session_id': request.session_id
                }).execute()

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Background generation error: {error_msg}")
        
        supabase.table('sessions').update({ 
            'generation_status': 'failed',
            'generation_error': error_msg
        }).eq('id', request.session_id).execute()

        # ERROR: Deduct 2000 tokens for Claude
        if request.model == 'claude-sonnet-4.5':
            supabase.rpc('deduct_tokens_with_tracking', {
                'p_user_id': request.user_id,
                'p_amount': 2000,
                'p_description': f'Error penalty - {request.model}',
                'p_session_id': request.session_id
            }).execute()


@app.post("/generate-async")
async def generate_code_async(request: AsyncGenerateRequest, background_tasks: BackgroundTasks):
    """
    Async endpoint - Returns immediately, processes in background
    Frontend listens to Supabase Realtime for status updates
    """
    background_tasks.add_task(process_generation_background, request)
    
    return JSONResponse({
        "success": True,
        "message": "Generation started in background",
        "session_id": request.session_id
    })


# ==================== STREAMING GENERATION (Keep for backward compatibility) ====================

async def generate_stream(request: GenerateRequest):
    """Stream generation progress to frontend with token tracking"""
    tokens_used = 0
    generation_success = False
    html_result = ""
    
    try:
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid model selected'})}\n\n"
            return
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing AI agent...'})}\n\n"
        await asyncio.sleep(0.3)
        
        if request.model == "claude-sonnet-4.5" and os.getenv("PERPLEXITY_API_KEY"):
            yield f"data: {json.dumps({'type': 'status', 'message': '🌐 Web search tools activated...'})}\n\n"
            await asyncio.sleep(0.4)
        
        yield f"data: {json.dumps({'type': 'status', 'message': '⚡ Generating code with AI...'})}\n\n"
        
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
        
        generation_success = True
        
        yield f"data: {json.dumps({'type': 'complete', 'html': html_result, 'total_tokens': tokens_used, 'token_usage': token_usage})}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Generation error: {error_msg}")
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'tokens_used': tokens_used})}\n\n"


@app.post("/generate")
async def generate_code_stream(request: GenerateRequest):
    """Streaming endpoint for code generation (backward compatible)"""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Garliq AI Agent Service on port {port}")
    print(f"🔑 Perplexity API: {'✅ Configured' if os.getenv('PERPLEXITY_API_KEY') else '❌ Not configured'}")
    print(f"🌐 Features: Real-time web search, Citations, Token tracking, Async processing")
    uvicorn.run(app, host="0.0.0.0", port=port)