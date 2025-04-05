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

# -a
english_to_urdu_agent = Agent(
    name="English_agent",
    handoff_description="AN english to urdu translator",
    instructions="You translate the user's message to Urdu "
)
urdu_to_english_agent = Agent(
    name="urdu_agent",
    handoff_description="AN urdu to English translator",
    instructions="You translate the user's message to english "
)

# -b
english_to_spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
)
spanish_to_english_agent = Agent(
    name="english_agent",
    instructions="You translate the user's message to english",
    handoff_description="A spanish to english translator",
)

# -c
english_to_arabic_agent = Agent(
    name="arabic_agent",
    handoff_description="An english to arabic translator",
    instructions="You translate the user's message to Arabic,and write it into arabic in right way carefuly"
)
arabic_to_english_agent = Agent(
    name="arabic_agent",
    handoff_description="An arabic to english translator",
    instructions="You translate the user's message to english."
)

# -d
english_to_german_agent = Agent(
    name="german_agent",
    handoff_description="An english to german translator",
    instructions="You translate the user's message to German"
)
german_to_english_agent = Agent(
    name="english_agent",
    handoff_description="A german to english translator",
    instructions="You translate the user's message to english"
)  

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[english_to_urdu_agent,urdu_to_english_agent,english_to_spanish_agent,spanish_to_english_agent,
              english_to_arabic_agent,arabic_to_english_agent,english_to_german_agent,]
)

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the English to Arabic,Urdu and spanish translator Agent. How i can help you today?").send()

# Step-6: Runner

@cl.on_message
async def handel_message(message: cl.Message):
    history =cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role":"user", "content":message.content})
    
    result = Runner.run_streamed(
        triage_agent,
        input=message.content,
        run_config=run_config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        
    history.append({"role":"assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    
    await cl.Message(content=result.final_output).send()


