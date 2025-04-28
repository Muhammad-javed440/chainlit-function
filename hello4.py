import os
import chainlit as cl
from agents import Agent,Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key=os.getenv("GEMINI_API_KEY")

# Step-1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# Step-2: Model
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash-exp",
    openai_client = provider,
)

# step-3: Config: Defined at Run Level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step-4: Agent
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

python_agent = Agent(
    name="Python Tutor",
    handoff_description="Expert in Python problem-solving and explanations.",
    instructions=(
        "Assist with Python-related questions. "
        "Explain your reasoning step by step and provide examples for clarity."
    )
)


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent,python_agent]
)

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! we are the python ,math and historical problems Support Agent. How we can help you today?").send()

# Step-6: Runner

@cl.on_message
async def handel_message(message: cl.Message):
    history =cl.user_session.get("history")
    
    msg = cl.Message(content="thinking......\n\n")
    await msg.send()
    
    history.append({"role":"user", "content":message.content})
    
    result = Runner.run_streamed(
        triage_agent ,
        input=history,
        run_config=run_config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        
    history.append({"role":"assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    
    


