import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from tools import save_text_to_file, search_tool, wikipedia_tool, terminal_run, file_read, file_write

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    filename: str


SYSTEM_PROMPT = """
/no_think
You are an assistant that helps with tasks on the system and occassionaly research tasks.

Work in steps:
1. If you need external facts, call the research tools available to you.
2. Read tool results in the conversation before deciding your next step.
3. If the user wants output persisted, call the appropriate tool to write results
   to disk before you finish.
4. If the user wants to read a file, call the appropriate tool to read the file.
5. If the user wants to write to a file, call the appropriate tool to write to the file.
6. If the user wants to run a terminal command, call the appropriate tool to run the command.
7. When all tasks are complete, return your structured response.

Choose tools from their descriptions and the user request. Do not invent tool usage.
In tools_used, list only tools you actually invoked, using their exact names.
Set filename to the path written when results were saved; otherwise use an empty string.
"""

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_ENDPOINT"),
    max_tokens=4096,
    temperature=0.6,
)

agent = create_agent(
    model=llm,
    tools=[search_tool, wikipedia_tool, save_text_to_file, terminal_run, file_read, file_write],
    system_prompt=SYSTEM_PROMPT,
    response_format=ResearchResponse,
    debug=True,
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            on_failure="continue",
        ),
    ],
)

query = input("What can I help you with? ")
result = agent.invoke({"messages": [HumanMessage(content=query)]})

structured_response = result.get("structured_response")
if structured_response:
    print(structured_response)
else:
    print("No structured response returned.")
    if result.get("messages"):
        print("Last message:", result["messages"][-1])
