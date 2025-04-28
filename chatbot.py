import os
import json
import chainlit as cl
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HISTORY_FILE = "chat_history.json"  # File to store history
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Step-1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step-2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=provider,
)

# Step-3: Config: Defined at Run Level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step-4: Agent
agent1 = Agent(
    name="Panaversity Support Agent",
    instructions="You are a helpful assistant that can answer the question"
)

# Function to load history from file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

# Function to save history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Step-5: Runner
@cl.on_chat_start
async def handle_chat_start():
    history = load_history()
    cl.user_session.set("history", history)
    await cl.Message(content="Thinking......\n\n").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    history.append({"role": "user", "content": message.content})
    
    result = await Runner.run(
        agent1,
        input=history,
        run_config=run_config
    )
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # Save history to file
    save_history(history)
    
  


