from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from agent import generate_initial_code, update_existing_code

load_dotenv()

app = FastAPI(title="Garliq AI Agent Service")

# CORS
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

@app.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    try:
        # Validate model
        if request.model not in ["llama-3.3-70b", "claude-sonnet-4.5"]:
            raise HTTPException(status_code=400, detail="Invalid model selected")
        
        # Generate code
        if request.current_code:
            # Update existing code
            html = update_existing_code(
                current_code=request.current_code,
                user_message=request.prompt,
                chat_history=[msg.dict() for msg in request.chat_history],
                model_name=request.model
            )
        else:
            # Generate initial code
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