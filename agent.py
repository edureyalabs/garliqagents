from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Dict, Any
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

# ==================== PERPLEXITY UNIFIED TOOL ====================

class PerplexitySearchInput(BaseModel):
    """Input schema for Perplexity AI search"""
    query: str = Field(..., description="The search query or research question")
    search_depth: str = Field(default="balanced", description="Search depth: 'quick' for fast results, 'balanced' for standard, or 'deep' for thorough research")

class PerplexityAITool(BaseTool):
    name: str = "perplexity_ai_search"
    description: str = """Ultimate AI-powered search and research tool with real-time web access.
    
    USE THIS TOOL FOR:
    - Current events, news, and real-time information
    - Market data, stock prices, crypto prices, exchange rates
    - Finding APIs, documentation, and technical resources
    - Research on any topic with citations
    - Getting statistics, facts, and verified information
    - Finding specific websites, URLs, or web content
    - Getting code examples and technical solutions
    
    Returns AI-generated answers with web sources and citations automatically."""
    args_schema: Type[BaseModel] = PerplexitySearchInput
    
    def _run(self, query: str, search_depth: str = "balanced") -> str:
        """Execute Perplexity AI search with built-in web search"""
        try:
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                return "❌ Perplexity API key not configured"
            
            url = "https://api.perplexity.ai/chat/completions"
            
            # Updated model selection based on search depth - using new model names
            model_map = {
                "quick": "sonar",  # Updated from llama-3.1-sonar-small-128k-online
                "balanced": "sonar",  # Updated from llama-3.1-sonar-large-128k-online
                "deep": "sonar-pro"  # Updated from llama-3.1-sonar-huge-128k-online
            }
            
            model = model_map.get(search_depth, "sonar")
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Provide accurate, detailed information with proper context. Focus on current, factual data."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract answer
            answer = data['choices'][0]['message']['content']
            
            # Extract citations (they are now returned by default in the response)
            citations = data.get('citations', [])
            
            # Format response
            result = f"🔍 SEARCH RESULTS:\n\n{answer}\n\n"
            
            if citations:
                result += "📚 SOURCES:\n"
                for idx, cite in enumerate(citations[:6], 1):
                    result += f"{idx}. {cite}\n"
            
            # Add usage info for debugging
            usage = data.get('usage', {})
            result += f"\n💡 Total tokens used: {usage.get('total_tokens', 'N/A')}"
            
            return result
            
        except requests.exceptions.Timeout:
            return "⏱️ Search timeout. The query took too long. Try a simpler search."
        except requests.exceptions.RequestException as e:
            return f"🌐 Search error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ==================== LLM CONFIGURATION ====================

def get_llm(model_name: str):
    """Get LLM based on model selection"""
    if model_name == "claude-sonnet-4.5":
        return LLM(
            model="anthropic/claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=50000
        )
    else:  # Default to Llama Groq
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=29000
        )


# ==================== AGENT CREATION ====================

def create_micro_app_agent(model_name: str, enable_tools: bool = True):
    """Create the world's most powerful micro-app generation agent"""
    llm = get_llm(model_name)
    
    # Perplexity tool only for Claude
    tools = []
    if enable_tools and model_name == "claude-sonnet-4.5":
        tools = [PerplexityAITool()]
    
    agent = Agent(
        role="Elite Micro-Application Architect & Real-Time Data Specialist",
        goal="""Create instant, powerful, single-page micro-applications that solve real human problems.
        Generate production-ready HTML/CSS/JavaScript applications that are:
        - Extremely fast to load and interact with
        - Visually stunning with modern UI/UX
        - Fully functional with real-time capabilities
        - Self-contained (no external dependencies)
        - Responsive and accessible
        - Enhanced with live data when needed""",
        
        backstory="""You are the world's most advanced micro-application architect, combining:

🌐 REAL-TIME DATA MASTERY: You have instant access to the world's knowledge through Perplexity AI,
which searches the entire web in real-time. Before coding, you research current data, market trends, 
API endpoints, documentation, and real-world information to make your applications truly intelligent 
and up-to-date. Perplexity automatically provides citations for all information.

💻 CODING EXCELLENCE: 15+ years of elite full-stack development experience. You write clean, 
performant, modern JavaScript/HTML/CSS that rivals professional development teams. You understand 
Single Page Architecture (SPA) patterns, state management, async operations, and real-time updates.

🎨 DESIGN BRILLIANCE: You create interfaces that make users say "WOW!" - smooth animations, 
beautiful gradients, thoughtful micro-interactions, and intuitive UX that feels native and modern.

⚡ MICRO-APP PHILOSOPHY: You specialize in instant-utility applications - not bloated software, 
but focused, powerful tools that do ONE thing exceptionally well. Think: live Bitcoin charts with 
price alerts, interactive stock dashboards, real-time weather visualizations, crypto trackers, 
data analytics tools, API integrations, research interfaces, and utilities that provide immediate value.

🔧 YOUR SUPERPOWERS:
- Research ANY topic using Perplexity (current events, APIs, data sources, documentation)
- Fetch live data from public APIs (stocks, crypto, weather, news, etc.)
- Create stunning interactive charts and visualizations (Canvas, SVG, CSS animations)
- Build real-time updating interfaces with WebSocket or polling
- Integrate with RESTful APIs and data sources
- Generate dynamic, data-driven content
- Craft responsive, mobile-first designs with accessibility

🎯 YOUR WORKFLOW:
1. **RESEARCH FIRST** (for data-driven apps): Use perplexity_ai_search to find:
   - Current API endpoints (free stock APIs, crypto APIs, weather APIs)
   - Latest data formats and response structures
   - Real-time data sources
   - Code examples and best practices
   
2. **DESIGN THE EXPERIENCE**: Think like a product designer:
   - What's the optimal user flow?
   - How can I make this delightful?
   - What micro-interactions will enhance usability?
   
3. **BUILD WITH EXCELLENCE**: Generate production-ready code:
   - Clean, semantic HTML5
   - Modern CSS with smooth animations
   - Robust JavaScript with error handling
   - Real-time data updates
   - Loading states and user feedback
   
4. **MAKE IT SHAREABLE**: Code should be self-contained and impressive

🌟 EXAMPLES OF YOUR WORK:
- Bitcoin price tracker with live chart and price alerts
- Stock market dashboard with multiple tickers
- Real-time weather app with beautiful animations
- Crypto portfolio tracker with gains/losses
- Interactive data visualizer
- API testing tool with JSON formatter
- Pomodoro timer with productivity stats

You don't just code - you create EXPERIENCES that users want to share.""",
        
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        max_iter=15
    )
    
    return agent


# ==================== CODE GENERATION FUNCTIONS ====================

def generate_initial_code(prompt: str, model_name: str) -> Dict[str, Any]:
    """Generate initial micro-app code from prompt with token tracking"""
    
    agent = create_micro_app_agent(model_name, enable_tools=(model_name == "claude-sonnet-4.5"))
    
    task_description = f"""
Create a complete, self-contained micro-application based on this request:

"{prompt}"

CRITICAL INSTRUCTIONS:

1. 🔍 RESEARCH PHASE (if needed):
   - If this requires REAL-TIME or CURRENT data (stocks, crypto, weather, news, statistics, APIs):
     * USE perplexity_ai_search tool FIRST to gather latest information
     * Search for: API endpoints, data formats, current values, documentation
     * Example searches: "free stock market API 2024", "Bitcoin price API", "weather API free"
   
   - For technical implementation questions:
     * Search for: "best way to create [feature] vanilla JavaScript"
     * Get code examples and best practices

2. 💻 CODE REQUIREMENTS:
   - Return ONLY pure HTML starting with <!DOCTYPE html>
   - Embed ALL CSS in <style> tags in <head>
   - Embed ALL JavaScript in <script> tags before </body>
   - Make it visually STUNNING - modern design that impresses
   - Use gradients, shadows, smooth animations, glass morphism
   - Ensure mobile responsiveness
   - Add loading states for async operations
   - Include proper error handling for API calls
   - Make it INTERACTIVE and ENGAGING

3. 🏗️ ARCHITECTURE:
   - Single Page Architecture (SPA) principles
   - Clean state management with vanilla JavaScript
   - Async/await for API calls with error handling
   - Event-driven updates
   - Performant rendering (use requestAnimationFrame for animations)

4. 📦 NO EXTERNAL DEPENDENCIES:
   - Use vanilla HTML/CSS/JS only
   - Can use public CDN libraries via <script> if absolutely needed (Chart.js, etc.)
   - Make it self-contained and portable

5. 📤 OUTPUT FORMAT:
   - NO markdown, NO code blocks, NO explanations
   - Just pure, clean HTML code
   - Start directly with: <!DOCTYPE html>

REMEMBER: Research first if needed, then build something AMAZING!
"""
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="Complete HTML code for a micro-application"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    
    # Extract HTML
    html = result.raw
    html = html.replace('```html\n', '').replace('```html', '').replace('```', '').strip()
    
    # Extract token usage - FIXED: Access attributes directly instead of using .get()
    token_usage_obj = result.token_usage
    
    # Create a dictionary with safe access
    if token_usage_obj:
        try:
            total_tokens = token_usage_obj.total_tokens if hasattr(token_usage_obj, 'total_tokens') else 0
            token_usage_dict = {
                'total_tokens': total_tokens,
                'prompt_tokens': token_usage_obj.prompt_tokens if hasattr(token_usage_obj, 'prompt_tokens') else 0,
                'completion_tokens': token_usage_obj.completion_tokens if hasattr(token_usage_obj, 'completion_tokens') else 0,
                'successful_requests': token_usage_obj.successful_requests if hasattr(token_usage_obj, 'successful_requests') else 0
            }
        except Exception as e:
            print(f"Warning: Could not extract token usage: {e}")
            total_tokens = 0
            token_usage_dict = {}
    else:
        total_tokens = 0
        token_usage_dict = {}
    
    return {
        'html': html,
        'total_tokens': total_tokens,
        'token_usage': token_usage_dict
    }


def update_existing_code(current_code: str, user_message: str, chat_history: list, model_name: str) -> Dict[str, Any]:
    """Update existing code based on user feedback with token tracking"""
    
    agent = create_micro_app_agent(model_name, enable_tools=(model_name == "claude-sonnet-4.5"))
    
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
    
    task_description = f"""
You are updating an existing micro-application based on user feedback.

CURRENT CODE (truncated):
{current_code[:3000]}...

PREVIOUS CONVERSATION:
{history_text}

USER'S NEW REQUEST:
"{user_message}"

INSTRUCTIONS:

1. 🔍 RESEARCH (if needed):
   - If the request requires NEW real-time data or external information:
     * Use perplexity_ai_search to gather current information
     * Find any new APIs or data sources needed

2. 💻 GENERATE UPDATED CODE:
   - Apply ALL requested changes
   - Maintain existing functionality unless asked to change
   - Keep the same quality and design standards
   - Ensure all features still work correctly

3. 📤 OUTPUT:
   - Return FULL updated HTML (not just changes)
   - Start with <!DOCTYPE html>
   - NO markdown, NO explanations
   - Just pure HTML code

Implement the changes now:
"""
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="Complete updated HTML code"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    
    # Extract HTML
    html = result.raw
    html = html.replace('```html\n', '').replace('```html', '').replace('```', '').strip()
    
    # Extract token usage - FIXED: Access attributes directly
    token_usage_obj = result.token_usage
    
    if token_usage_obj:
        try:
            total_tokens = token_usage_obj.total_tokens if hasattr(token_usage_obj, 'total_tokens') else 0
            token_usage_dict = {
                'total_tokens': total_tokens,
                'prompt_tokens': token_usage_obj.prompt_tokens if hasattr(token_usage_obj, 'prompt_tokens') else 0,
                'completion_tokens': token_usage_obj.completion_tokens if hasattr(token_usage_obj, 'completion_tokens') else 0,
                'successful_requests': token_usage_obj.successful_requests if hasattr(token_usage_obj, 'successful_requests') else 0
            }
        except Exception as e:
            print(f"Warning: Could not extract token usage: {e}")
            total_tokens = 0
            token_usage_dict = {}
    else:
        total_tokens = 0
        token_usage_dict = {}
    
    return {
        'html': html,
        'total_tokens': total_tokens,
        'token_usage': token_usage_dict
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    test_prompt = "Create a live Bitcoin price tracker with a beautiful chart"
    model_name = "claude-sonnet-4.5"
    
    try:
        result = generate_initial_code(test_prompt, model_name)
        print("Generated HTML Code:\n", result['html'][:500])
        print(f"\nTokens Used: {result['total_tokens']}")
        print(f"Token Breakdown: {result['token_usage']}")
    except Exception as e:
        print("Error during test generation:", e)
        import traceback
        traceback.print_exc()