import os
import requests
from bs4 import BeautifulSoup
import wikipedia
import numexpr
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from tavily import TavilyClient

load_dotenv()

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = "us-east-1"


def tavily_search(query: str) -> str:
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No results found."
        return "\n\n".join(f"{r['title']}\n{r['content']}" for r in results)
    except Exception as e:
        return f"Search unavailable: {e}"


def calculate(expression: str) -> str:
    try:
        result = numexpr.evaluate(expression.strip())
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_weather(location: str) -> str:
    try:
        url = f"https://wttr.in/{requests.utils.quote(location)}?format=3"
        response = requests.get(url, timeout=10)
        return response.text.strip()
    except Exception as e:
        return f"Weather unavailable: {e}"


def read_url(url: str) -> str:
    try:
        response = requests.get(url.strip(), timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Trim to avoid overwhelming the context
        return text[:3000] + ("..." if len(text) > 3000 else "")
    except Exception as e:
        return f"Could not read URL: {e}"


def wiki_search(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=5)
    except Exception as e:
        return f"Wikipedia unavailable: {e}"


def build_agent() -> AgentExecutor:
    llm = ChatBedrock(
        model_id=MODEL_ID,
        region_name=REGION,
        model_kwargs={"max_tokens": 2048},
    )

    tools = [
        Tool(
            name="web_search",
            func=tavily_search,
            description=(
                "Search the web for current information. "
                "Use this when you need facts, news, or data you don't know."
            ),
        ),
        Tool(
            name="calculator",
            func=calculate,
            description=(
                "Evaluate a math expression. "
                "Input must be a valid numeric expression like '2 ** 10' or '(3 + 5) * 12'. "
                "Use this for any arithmetic or math calculations."
            ),
        ),
        Tool(
            name="weather",
            func=get_weather,
            description=(
                "Get the current weather for a location. "
                "Input should be a city name or city and country, e.g. 'London' or 'Tokyo, Japan'."
            ),
        ),
        Tool(
            name="read_url",
            func=read_url,
            description=(
                "Fetch and read the text content of a webpage given its URL. "
                "Use this to read articles, documentation, or any specific webpage."
            ),
        ),
        Tool(
            name="wikipedia",
            func=wiki_search,
            description=(
                "Look up a topic on Wikipedia for factual, encyclopedic information. "
                "Use this for definitions, historical facts, or background on a topic."
            ),
        ),
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)


def main():
    executor = build_agent()
    print("Web Research Agent ready. Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = executor.invoke({"input": question})
        print(f"\nAnswer: {result['output']}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
