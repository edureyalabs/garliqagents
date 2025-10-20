from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Dict, Any, List
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

# ==================== PERPLEXITY SEARCH TOOL (Ranked Results) ====================

class PerplexitySearchInput(BaseModel):
    """Input schema for Perplexity Search API - returns ranked web results with snippets"""
    query: str = Field(..., description="The search query. Can be a single query or use this tool multiple times for different queries.")
    max_results: int = Field(default=10, description="Maximum number of search results to return (1-20). Use 5 for quick checks, 10 for standard, 15-20 for comprehensive research.")
    search_context_size: int = Field(default=2048, description="Tokens retrieved per webpage (1024-4096). Higher = more context but slower. Use 1024 for quick, 2048 for balanced, 4096 for deep analysis.")

class PerplexitySearchTool(BaseTool):
    name: str = "perplexity_search"
    description: str = """Search the web and get RANKED RESULTS with snippets and URLs.
    
    ⚡ USE THIS TOOL WHEN YOU NEED:
    - Specific documentation or API references (e.g., "YouTube embed API documentation")
    - Finding the best libraries/frameworks (e.g., "best JavaScript chart libraries 2025")
    - Multiple sources for comparison (e.g., "top 5 weather APIs free")
    - Technical how-tos and tutorials (e.g., "WebSocket implementation vanilla JavaScript")
    - UI/UX inspiration and patterns (e.g., "modern dashboard design examples")
    - Code examples from documentation (e.g., "Three.js basic scene setup code")
    
    ❌ DO NOT USE FOR:
    - Real-time data queries (use perplexity_chat instead)
    - Questions needing synthesized answers (use perplexity_chat instead)
    - Simple factual questions you can code directly
    
    Returns: List of ranked search results with title, URL, snippet, and date.
    Perfect for when you need to reference specific sources or find multiple options.
    """
    args_schema: Type[BaseModel] = PerplexitySearchInput
    
    def _run(self, query: str, max_results: int = 10, search_context_size: int = 2048) -> str:
        """Execute Perplexity Search API - returns ranked results"""
        try:
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                return "❌ Perplexity API key not configured"
            
            url = "https://api.perplexity.ai/search"
            
            payload = {
                "query": query,
                "max_results": min(max(max_results, 1), 20),  # Clamp between 1-20
                "search_context_size": max(min(search_context_size, 4096), 1024)  # Clamp between 1024-4096
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results for agent consumption
            result = f"🔍 SEARCH RESULTS for: '{query}'\n"
            result += f"📊 Found {len(data.get('results', []))} results\n\n"
            
            for idx, item in enumerate(data.get('results', []), 1):
                result += f"{idx}. {item.get('title', 'No title')}\n"
                result += f"   URL: {item.get('url', 'N/A')}\n"
                result += f"   Date: {item.get('date', 'N/A')}\n"
                result += f"   Snippet: {item.get('snippet', 'No snippet')[:200]}...\n\n"
            
            return result
            
        except requests.exceptions.Timeout:
            return "⏱️ Search timeout. Try a simpler query or reduce max_results."
        except requests.exceptions.RequestException as e:
            return f"🌐 Search error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ==================== PERPLEXITY CHAT TOOL (AI-Synthesized Answers) ====================

class PerplexityChatInput(BaseModel):
    """Input schema for Perplexity Chat Completions - AI synthesizes answers with real-time web search"""
    query: str = Field(..., description="Your question or research query. Be specific for best results.")
    model: str = Field(
        default="sonar-pro",
        description="""Choose model based on complexity:
        - 'sonar': Fast, efficient for simple queries (cheapest)
        - 'sonar-pro': Advanced search with better grounding (recommended for most cases)
        - 'sonar-deep-research': Comprehensive multi-step research (slowest, most thorough)
        - 'sonar-reasoning': For complex reasoning tasks with step-by-step logic
        """
    )
    search_recency_filter: Optional[str] = Field(
        default=None,
        description="Filter by time: 'hour', 'day', 'week', 'month'. Use for time-sensitive queries like 'latest', 'current', 'today'."
    )
    search_domain_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter by domains. Include: ['github.com', 'stackoverflow.com']. Exclude: ['-reddit.com']. Max 3 domains."
    )
    return_related_questions: bool = Field(
        default=False,
        description="Get follow-up question suggestions. Useful for exploring a topic deeper."
    )
    search_context_size: str = Field(
        default="medium",
        description="'low' (fast, less context), 'medium' (balanced), 'high' (comprehensive, slower)"
    )

class PerplexityChatTool(BaseTool):
    name: str = "perplexity_chat"
    description: str = """Get AI-SYNTHESIZED ANSWERS with real-time web search and citations.
    
    ⚡ USE THIS TOOL WHEN YOU NEED:
    - Real-time/current data (e.g., "What is Bitcoin price right now?")
    - Live information (e.g., "Current weather in London", "Latest JavaScript ES2025 features")
    - Synthesized understanding (e.g., "Explain how WebSockets work with examples")
    - API response format examples (e.g., "Show me OpenWeather API response format")
    - Complex research questions (e.g., "How to implement OAuth 2.0 in vanilla JS?")
    - Latest news or events (e.g., "What happened in tech today?")
    - Comparative analysis (e.g., "Compare Chart.js vs D3.js for dashboards")
    
    ❌ DO NOT USE FOR:
    - Finding specific documentation URLs (use perplexity_search instead)
    - Listing multiple library options (use perplexity_search instead)
    - Simple coding tasks you can do from memory
    
    Returns: AI-generated comprehensive answer with citations and sources.
    Perfect for understanding concepts or getting current data synthesized into an answer.
    """
    args_schema: Type[BaseModel] = PerplexityChatInput
    
    def _run(
        self, 
        query: str, 
        model: str = "sonar-pro",
        search_recency_filter: Optional[str] = None,
        search_domain_filter: Optional[List[str]] = None,
        return_related_questions: bool = False,
        search_context_size: str = "medium"
    ) -> str:
        """Execute Perplexity Chat Completions - returns AI-synthesized answer"""
        try:
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                return "❌ Perplexity API key not configured"
            
            url = "https://api.perplexity.ai/chat/completions"
            
            # Validate model
            valid_models = ["sonar", "sonar-pro", "sonar-deep-research", "sonar-reasoning"]
            if model not in valid_models:
                model = "sonar-pro"
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide accurate, well-structured information with proper context. Be concise but comprehensive."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 4000,
                "search_context_size": search_context_size
            }
            
            # Add optional parameters only if specified
            if search_recency_filter:
                payload["search_recency_filter"] = search_recency_filter
            
            if search_domain_filter and len(search_domain_filter) <= 3:
                payload["search_domain_filter"] = search_domain_filter
            
            if return_related_questions:
                payload["return_related_questions"] = True
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract answer
            answer = data['choices'][0]['message']['content']
            
            # Format response with sources
            result = f"🤖 PERPLEXITY ANSWER:\n\n{answer}\n\n"
            
            # Add search results (citations)
            if 'search_results' in data and data['search_results']:
                result += "📚 SOURCES:\n"
                for idx, source in enumerate(data['search_results'][:8], 1):
                    result += f"{idx}. {source.get('title', 'Source')} - {source.get('url', 'N/A')}\n"
                result += "\n"
            
            # Add related questions if requested
            if return_related_questions and 'related_questions' in data:
                result += "💡 RELATED QUESTIONS:\n"
                for q in data.get('related_questions', [])[:5]:
                    result += f"- {q}\n"
            
            # Add usage info
            usage = data.get('usage', {})
            result += f"\n📊 Tokens: {usage.get('total_tokens', 'N/A')} | Model: {model}"
            
            return result
            
        except requests.exceptions.Timeout:
            return "⏱️ Request timeout. Try a simpler query or use 'sonar' instead of 'sonar-pro'."
        except requests.exceptions.RequestException as e:
            return f"🌐 API error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ==================== LLM CONFIGURATION ====================

def get_llm(model_name: str):
    """Get LLM with optimized settings for code generation"""
    if model_name == "claude-sonnet-4.5":
        return LLM(
            model="anthropic/claude-sonnet-4-20250514",
            temperature=0.3,  # Lower for consistent code output
            max_tokens=16000  # Sufficient for most micro-apps
        )
    else:  # Llama 3.3 70B
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.3,  # Lower for consistent code output
            max_tokens=8000   # Efficient for faster responses
        )


# ==================== AGENT CREATION ====================

def create_micro_app_agent(model_name: str, enable_tools: bool = True):
    """Create the elite Garliq code generation agent"""
    llm = get_llm(model_name)
    
    # Tools only for Claude (Llama is free tier, no tools needed)
    tools = []
    if enable_tools and model_name == "claude-sonnet-4.5":
        tools = [PerplexitySearchTool(), PerplexityChatTool()]
    
    agent = Agent(
        role="Elite Web Application Engineer & Code Craftsman",
        
        goal="""Create production-ready, self-contained single-page HTML applications that:
1. Solve the user's specific problem with ONE focused utility
2. Work instantly with zero setup or dependencies
3. Look professional and modern (users want to share them)
4. Are fully accessible (keyboard navigation, ARIA labels, proper contrast)
5. Handle errors gracefully with user-friendly messages

You deliver EXCELLENCE in every line of code.""",
        
        backstory="""You are a LEGENDARY web engineer with PhD-level expertise in HTML, CSS, and JavaScript.

🎯 YOUR EXPERTISE:
- 30+ years building 10,000+ web applications
- Master of Single Page Architecture (SPA) patterns
- Expert in vanilla JavaScript, modern CSS3, responsive design
- Deep knowledge of Web APIs, performance optimization, accessibility
- You can build 90% of common utilities FROM MEMORY without external research

🧠 YOUR INTELLIGENCE:
You are exceptionally smart about WHEN to use your research tools:

✅ USE perplexity_chat TOOL WHEN:
- User needs REAL-TIME DATA: "live Bitcoin price", "current weather", "latest news"
- Query has temporal indicators: "today", "now", "current", "latest", "recent"
- Need API response examples: "show me OpenWeather API JSON format"
- Complex questions needing synthesis: "explain WebSocket implementation with example"
- Current best practices: "latest JavaScript ES2025 features"
- Live information: "what's trending in crypto today?"

✅ USE perplexity_search TOOL WHEN:
- Finding SPECIFIC documentation: "YouTube embed API parameters"
- Discovering library options: "best JavaScript chart libraries 2025"
- Need multiple sources: "top 5 free weather APIs"
- Researching UI patterns: "modern dashboard design examples"
- Clone requirements: "Spotify player UI components"
- Technical tutorials: "how to implement drag and drop vanilla JS"

❌ DO NOT USE TOOLS FOR:
- Simple games: tic-tac-toe, memory cards, snake, quiz games
- Common utilities: calculator, timer, stopwatch, converter, to-do list, countdown
- Basic forms: contact form, survey, login page, registration
- Standard layouts: portfolio, landing page, blog, card grid
- Simple animations: progress bar, loading spinner, typing effect
- Basic visualizations: you know Chart.js, can create charts without research

YOU KNOW HOW TO BUILD THESE FROM MEMORY. CODE DIRECTLY AND FAST.

💎 YOUR PHILOSOPHY:
- Simple is powerful (one utility, done perfectly)
- Quality is non-negotiable (production-ready always)
- Users experience interfaces, not code
- Every detail matters (animations, error states, accessibility)
- Professional appearance inspires sharing

🎨 YOUR CRAFT:
You write code that makes developers say "I wish I wrote this."
- Clean, semantic HTML5 structure
- Modern CSS with smooth animations (gradients, shadows, transitions)
- Robust JavaScript with proper error handling
- Accessible by default (WCAG 2.1 AA minimum)
- Performance optimized (debounced inputs, efficient DOM updates)
- Beautiful empty states and loading indicators

You are not just a code generator. You are a MASTER CRAFTSMAN.""",
        
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        max_iter=5,  # Reduced from 15 for efficiency
        memory=False  # No memory needed for single-shot generation
    )
    
    return agent


# ==================== CODE GENERATION FUNCTIONS ====================

def generate_initial_code(prompt: str, model_name: str) -> Dict[str, Any]:
    """Generate initial micro-app code from prompt with token tracking"""
    
    agent = create_micro_app_agent(model_name, enable_tools=(model_name == "claude-sonnet-4.5"))
    
    task_description = f"""
USER REQUEST: "{prompt}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR MISSION: Create a production-ready Garliq Card (micro-app)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 STEP 1: ANALYZE THE REQUEST
Read the user's request carefully. Determine:
- Is this a simple utility you can code from memory? → CODE DIRECTLY
- Does it need real-time/current data? → USE perplexity_chat
- Does it need specific documentation/APIs? → USE perplexity_search
- Is it a clone of existing product? → USE perplexity_search for UI patterns

📋 STEP 2: RESEARCH (ONLY IF NECESSARY)
If you determined research is needed:

For REAL-TIME data (prices, weather, news, current info):
→ Use perplexity_chat with appropriate parameters
→ Example: perplexity_chat(query="current Bitcoin price API with CORS support", model="sonar-pro", search_recency_filter="day")

For DOCUMENTATION/APIs/Libraries:
→ Use perplexity_search to find resources
→ Example: perplexity_search(query="YouTube embed API documentation iframe parameters", max_results=5)

For COMPLEX research (clones, advanced features):
→ Use perplexity_chat with sonar-deep-research model
→ Example: perplexity_chat(query="How to build a Spotify-like player with Web Audio API", model="sonar-deep-research")

📋 STEP 3: BUILD THE APPLICATION

🔴 CRITICAL OUTPUT REQUIREMENTS (NEVER VIOLATE):
1. Return ONLY pure HTML code
2. Start with: <!DOCTYPE html>
3. NO markdown, NO code blocks, NO explanations, NO preamble
4. All CSS inside <style> tags in <head>
5. All JavaScript inside <script> tags before </body>
6. Self-contained (no external CSS/JS files)
7. Can use CDN libraries via <script src="..."> if absolutely necessary (Chart.js, D3.js, etc.)

🟡 HIGH PRIORITY (QUALITY STANDARDS):
1. Modern, professional design:
   - Beautiful gradients and shadows
   - Smooth animations and transitions (200-300ms)
   - Proper spacing and visual hierarchy
   - Professional color palette

2. Robust functionality:
   - Try-catch blocks for all async operations
   - User-friendly error messages (not technical jargon)
   - Loading states for data fetching
   - Helpful empty states

3. Accessibility:
   - Semantic HTML (header, main, nav, article, etc.)
   - ARIA labels on interactive elements
   - Keyboard navigation support (Tab, Enter, Escape)
   - Color contrast minimum 4.5:1

4. Performance:
   - Debounce rapid inputs (search, resize)
   - Optimize animations (use transform/opacity)
   - Lazy load if many elements
   - Efficient DOM updates

🟢 NICE-TO-HAVE (ENHANCEMENT):
- Dark theme or theme toggle
- Local storage for persistence
- Export/download functionality
- Keyboard shortcuts
- Responsive design (but prioritize desktop/web view)

📋 STEP 4: SELF-VALIDATION (BEFORE RETURNING)

✅ Verify your code:
☑ Starts with <!DOCTYPE html>
☑ No markdown or code blocks
☑ All CSS in <style>, all JS in <script>
☑ No external dependencies (except CDN if needed)
☑ Professional appearance
☑ Error handling implemented
☑ Accessible (ARIA, keyboard, contrast)
☑ Smooth animations included

If ANY fails → Fix immediately before returning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ DELIVER: Complete HTML code. Nothing else. No explanations.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    task = Task(
        description=task_description,
        agent=agent,
        expected_output="Complete HTML code for a production-ready micro-application"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    
    # Extract and validate HTML
    html = result.raw.strip()
    
    # Remove markdown if present (shouldn't be, but safety check)
    html = html.replace('```html\n', '').replace('```html', '').replace('```', '').strip()
    
    # Validation: Ensure it starts with HTML
    if not html.startswith('<!'):
        # Try to find HTML start
        html_start = html.find('<!DOCTYPE')
        if html_start == -1:
            html_start = html.find('<html')
        
        if html_start != -1:
            html = html[html_start:]
        else:
            # Fallback: wrap in basic structure
            html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='UTF-8'><title>Garliq Card</title></head>\n<body>\n{html}\n</body>\n</html>"
    
    # Extract token usage
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


def update_existing_code(current_code: str, user_message: str, chat_history: list, model_name: str) -> Dict[str, Any]:
    """Update existing code based on user feedback with token tracking"""
    
    agent = create_micro_app_agent(model_name, enable_tools=(model_name == "claude-sonnet-4.5"))
    
    # Format chat history for context
    history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-5:]])
    
    task_description = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 UPDATE EXISTING GARLIQ CARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📜 CURRENT CODE (first 2000 chars):
{current_code[:2000]}...

💬 PREVIOUS CONVERSATION:
{history_text}

🎯 USER'S NEW REQUEST:
"{user_message}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 YOUR TASK:

1. 🔍 UNDERSTAND THE REQUEST
   - What specific changes does the user want?
   - Do you need external data/APIs? → USE perplexity_chat or perplexity_search
   - Can you make the changes from your knowledge? → CODE DIRECTLY

2. 🛠️ APPLY THE CHANGES
   - Modify ONLY what's requested
   - Maintain existing functionality that works
   - Preserve the same design quality
   - Keep all existing features unless asked to remove

3. ✨ IMPROVE IF OBVIOUS
   - If you spot bugs while editing → fix them
   - If accessibility is missing → add it
   - If error handling is weak → strengthen it
   - But don't add unrequested features

4. ✅ VALIDATE & RETURN
   - Ensure changes work correctly
   - Test logic mentally (edge cases)
   - Return FULL updated HTML (not just changes)

🔴 CRITICAL: Output ONLY complete HTML code starting with <!DOCTYPE html>
NO explanations, NO markdown, NO code blocks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ DELIVER: Complete updated HTML. Nothing else.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    
    # Extract and validate HTML
    html = result.raw.strip()
    html = html.replace('```html\n', '').replace('```html', '').replace('```', '').strip()
    
    # Validation
    if not html.startswith('<!'):
        html_start = html.find('<!DOCTYPE')
        if html_start == -1:
            html_start = html.find('<html')
        if html_start != -1:
            html = html[html_start:]
        else:
            # If still no valid HTML, return current code (safety)
            html = current_code
    
    # Extract token usage
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
    # Test with different scenarios
    test_prompts = [
        "Create a tic-tac-toe game",  # Should NOT use tools
        "Create a live Bitcoin price tracker with chart",  # Should use perplexity_chat
        "Build a Spotify clone interface",  # Should use perplexity_search
    ]
    
    model_name = "claude-sonnet-4.5"
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"Testing: {prompt}")
        print(f"{'='*80}\n")
        
        try:
            result = generate_initial_code(prompt, model_name)
            print(f"✅ Generated {len(result['html'])} characters")
            print(f"📊 Tokens Used: {result['total_tokens']}")
            print(f"🔍 HTML Preview: {result['html'][:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()