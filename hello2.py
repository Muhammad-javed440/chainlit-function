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
fiverr_gig_agent = Agent(
    name="fiverr gig creator",
    handoff_description="Specialist Agent for fiverr gig creator",
    instructions="You provide help with fiverr gig creator. Explain your reasoning at each step and include examples"
)

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the fiverr gig create Support Agent. How i can help you today?").send()

# Step-6: Runner

@cl.on_message
async def handel_message(message: cl.Message):
    history =cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role":"user", "content":message.content})
    
    result = Runner.run_streamed(
        fiverr_gig_agent,
        input=message.content,
        run_config=run_config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        
    history.append({"role":"assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    
    await cl.Message(content=result.final_output).send()


