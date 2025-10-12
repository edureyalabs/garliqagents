from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import asyncio
from dotenv import load_dotenv
from agent import generate_initial_code, update_existing_code

load_dotenv()

app = FastAPI(title="Garliq AI Agent Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama-3.3-70b"
    current_code: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []

class GenerateResponse(BaseModel):
    html: str
    success: bool
    error: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "Garliq AI Agent Service Running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

async def generate_stream(request: GenerateRequest):
    """Stream generation progress to frontend"""
    try:
        # Validate model
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid model selected'})}\n\n"
            return
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing AI agent...'})}\n\n"
        await asyncio.sleep(0.3)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your prompt...'})}\n\n"
        await asyncio.sleep(0.5)
        
        if request.current_code:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Reading current code structure...'})}\n\n"
            await asyncio.sleep(0.4)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Planning modifications...'})}\n\n"
            await asyncio.sleep(0.3)
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Designing page structure...'})}\n\n"
            await asyncio.sleep(0.4)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Writing HTML structure...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Crafting CSS styles...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Adding JavaScript interactions...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating code with AI...'})}\n\n"
        
        # Generate code
        if request.current_code:
            html = update_existing_code(
                current_code=request.current_code,
                user_message=request.prompt,
                chat_history=[msg.dict() for msg in request.chat_history],
                model_name=request.model
            )
        else:
            html = generate_initial_code(
                prompt=request.prompt,
                model_name=request.model
            )
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Finalizing code...'})}\n\n"
        await asyncio.sleep(0.3)
        
        # Send complete HTML
        yield f"data: {json.dumps({'type': 'complete', 'html': html})}\n\n"
        
    except Exception as e:
        print(f"Stream error: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

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

# Fallback non-streaming endpoint (for compatibility)
@app.post("/generate-sync", response_model=GenerateResponse)
async def generate_code_sync(request: GenerateRequest):
    try:
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            raise HTTPException(status_code=400, detail="Invalid model selected")
        
        if request.current_code:
            html = update_existing_code(
                current_code=request.current_code,
                user_message=request.prompt,
                chat_history=[msg.dict() for msg in request.chat_history],
                model_name=request.model
            )
        else:
            html = generate_initial_code(
                prompt=request.prompt,
                model_name=request.model
            )
        
        return GenerateResponse(html=html, success=True)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return GenerateResponse(
            html="",
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)