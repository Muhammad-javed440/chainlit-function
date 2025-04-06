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
urdu_agent = Agent(
    name="English_agent",
    handoff_description="Any language to urdu translator",
    instructions="You translate the user's message to Urdu "
)

# -b
spanish_agent = Agent(
    name="spanish_agent",
    handoff_description="Any language to spanish translator",
     instructions="You translate the user's message to Spanish"
)

# -c
arabic_agent = Agent(
    name="arabic_agent",
    handoff_description="Any language to arabic translator",
    instructions="You translate the user's message to Arabic,and write it into arabic in right way carefuly"
)

# -d
german_agent = Agent(
    name="german_agent",
    handoff_description="Any language to german translator",
    instructions="You translate the user's message to German"
)

# -E
hindi_agent = Agent(
    name="hindi_agent",
    handoff_description="Any language to hindi translator",
    instructions="You translate the user's message to hindi"
)

# -F
russian_agent = Agent(
    name="russian_agent",
    handoff_description="Any language to russian translator",
    instructions="You translate the user's message to russian"
)

# -G
italian_agent = Agent(
    name="italian_agent",
    handoff_description="Any language to italian translator",
    instructions="You translate the user's message to italian"
)

# -h
english_agent = Agent(
    name="urdu_agent",
    handoff_description="Any language to English translator",
    instructions="You translate the user's message to english "
)

# -I
chinies_agent = Agent(
    name="chinies_agent",
    handoff_description="Any language to chinies translator",
    instructions="You translate the user's message to chinies "
)

# -J
astrailian_agent = Agent(
    name="astrailian_agent",
    handoff_description="Any language to astrailian translator",
    instructions="You translate the user's message to astrailian "
)


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[urdu_agent,english_agent,spanish_agent,arabic_agent,german_agent,hindi_agent,russian_agent,italian_agent,chinies_agent,astrailian_agent]
)

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the English to Arabic, Urdu, hindi, italian, russian, german and spanish translator Agent. How i can help you today?").send()

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


