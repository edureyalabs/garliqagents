from crewai import Agent, LLM
# from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env


def get_llm(model_name: str):
    """Get LLM based on model selection"""
    if model_name == "claude-sonnet-4.5":
        return LLM(
            model="anthropic/claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=4000
        )
    else:  # Default to Llama Groq
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=4000
        )


def create_coding_agent(model_name: str):
    """Create a single powerful coding agent"""
    llm = get_llm(model_name)

    agent = Agent(
        role="Expert Full-Stack Web Developer",
        goal="Generate complete, production-ready HTML/CSS/JavaScript code based on user requirements",
        backstory="""You are an elite web developer with 15+ years of experience. 
You specialize in creating beautiful, modern, responsive web applications.
You write clean, semantic HTML with embedded CSS and JavaScript.
You always follow best practices and modern web standards.
You create visually stunning interfaces with smooth animations and interactions.""",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    return agent


def generate_initial_code(prompt: str, model_name: str) -> str:
    """Generate initial HTML code from prompt"""
    agent = create_coding_agent(model_name)

    task_description = f"""
Create a complete, self-contained HTML page based on this request:

"{prompt}"

Requirements:
- Return ONLY the HTML code starting with <!DOCTYPE html>
- Embed all CSS in <style> tags in the <head>
- Embed all JavaScript in <script> tags before </body>
- Make it visually appealing with modern design
- Use inline styles or embedded CSS (no external files)
- Ensure it's fully functional and interactive
- NO markdown formatting, NO explanations, NO code blocks
- Just pure HTML code

Generate the complete HTML now:
"""

    result = agent.kickoff(task_description)

    html = result.raw
    html = html.replace('```html\n', '').replace('``````', '').strip()

    return html


def update_existing_code(current_code: str, user_message: str, chat_history: list, model_name: str) -> str:
    """Update existing code based on user feedback"""
    agent = create_coding_agent(model_name)

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

    task_description = f"""
You are updating existing HTML code based on user feedback.

CURRENT CODE:
{current_code[:3000]}...

PREVIOUS CONVERSATION:
{history_text}

USER'S NEW REQUEST:
"{user_message}"

Generate the COMPLETE UPDATED HTML code:
- Return the FULL page with all updates applied
- Keep existing functionality unless asked to change it
- Only return HTML code, no explanations
- Start with <!DOCTYPE html>
"""

    result = agent.kickoff(task_description)

    html = result.raw
    html = html.replace('```html\n', '').replace('``````', '').strip()

    return html


if __name__ == "__main__":
    test_prompt = "Create a simple hello world HTML page"
    model_name = "llama-3.3-70b"
    try:
        html_code = generate_initial_code(test_prompt, model_name)
        print("Generated HTML Code:\n", html_code)
    except Exception as e:
        print("Error during test generation:", e)
