import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    filename: str

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_ENDPOINT"),
    max_tokens=4096,
    temperature=0.6,
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    /no_think
    You are a research assistant that helps generate research papers.

    Work in steps:
    1. If you need external facts, call the research tools available to you.
    2. Read tool results in the conversation before deciding your next step.
    3. If the user wants output persisted, call the appropriate tool to write results
       to disk before you finish.
    4. When research (and any persistence) is complete, respond with your final message
       as ONLY valid JSON matching the schema below. Never send an empty final message.

    Choose tools from their descriptions and the user request. Do not invent tool usage.
    In tools_used, list only tools you actually invoked, using their exact names.
    Set filename to the path written when results were saved; otherwise use an empty string.

    {format_instructions}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
query = input("What can i help you research?")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

def extract_output_text(output) -> str:
    if output is None:
        return ""
    if isinstance(output, list):
        return output[0].get("text", "") if output else ""
    return output

try:
    text = extract_output_text(raw_response.get("output")).strip()
    if not text:
        print("Agent returned empty output. Intermediate steps:")
        for action, observation in raw_response.get("intermediate_steps", []):
            print(f"  tool={action.tool}, input={action.tool_input}")
            print(f"  observation={str(observation)[:300]}...")
        raise ValueError("Empty agent output — model did not produce a final JSON response.")

    data = json.loads(text)
    structured_response = ResearchResponse.model_validate(data)
    print(structured_response)
except json.JSONDecodeError as e:
    print("Error parsing JSON", e, "Raw Response", raw_response)
except ValidationError as e:
    print("Error validating response schema", e, "Raw Response", raw_response)
except ValueError as e:
    print(e)
