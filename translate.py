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
japanese_agent = Agent(
    name="japanese_agent",
    handoff_description="Any language to japanese translator",
    instructions="You translate the user's message to japanese "
)

# -k
bangali_agent = Agent(
    name="bangali_agent",
    handoff_description="Any language to bangali translator",
    instructions="You translate the user's message to bangali "
)


# -L
nepali_agent = Agent(
    name="astrailian_agent",
    handoff_description="Any language to nepali translator",
    instructions="You translate the user's message to nepali "
)

# -M
punjabi_agent = Agent(
    name="punjabi_agent",
    handoff_description="Any language to punjabi translator",
    instructions="You translate the user's message to punjabi "
)

# -N
korean_agent = Agent(
    name="korean_agent",
    handoff_description="Any language to korean translator",
    instructions="You translate the user's message to korean "
)

# -O
albanian_agent = Agent(
    name="albanian_agent",
    handoff_description="Any language to albanian translator",
    instructions="You translate the user's message to albanian "
)

# -P
turkish_agent = Agent(
    name="turkish_agent",
    handoff_description="Any language to turkish translator",
    instructions="You translate the user's message to turkish "
)

# -Q
pashto_agent = Agent(
    name="pashto_agent",
    handoff_description="Any language to pashto translator",
    instructions="You translate the user's message to pashto "
)

# -R
gujarati_agent = Agent(
    name="gujarati_agent",
    handoff_description="Any language to gujarati translator",
    instructions="You translate the user's message to gujarati "
)

# -S
french_agent = Agent(
    name="french_agent",
    handoff_description="Any language to french translator",
    instructions="You translate the user's message to french "
)

# -T
latin_agent = Agent(
    name="latin_agent",
    handoff_description="Any language to latin translator",
    instructions="You translate the user's message to latin "
)

# -U
persian_agent = Agent(
    name="persian_agent",
    handoff_description="Any language to persian/farsi translator",
    instructions="You translate the user's message to persian/farsi "
)

# -V
filipino_agent = Agent(
    name="afilipino_agent",
    handoff_description="Any language to filipino translator",
    instructions="You translate the user's message to filipino "
)

# -W
danish_agent = Agent(
    name="danish_agent",
    handoff_description="Any language to danish translator",
    instructions="You translate the user's message to danish "
)

# -X
dari_agent = Agent(
    name="adari_agent",
    handoff_description="Any language to dari translator",
    instructions="You translate the user's message to dari "
)

# -Y
bosnian_agent = Agent(
    name="bosnian_agent",
    handoff_description="Any language to bosnian translator",
    instructions="You translate the user's message to bosnian "
)

# -Z
catalan_agent = Agent(
    name="catalan_agent",
    handoff_description="Any language to catalan translator",
    instructions="You translate the user's message to catalan "
)

# -ab
croatian_agent = Agent(
    name="croatian_agent",
    handoff_description="Any language to croatian translator",
    instructions="You translate the user's message to croatian "
)

# -ac
finnish_agent = Agent(
    name="finnish_agent",
    handoff_description="Any language to finnish translator",
    instructions="You translate the user's message to finnish "
)

# -ad
georgian_agent = Agent(
    name="georgian_agent",
    handoff_description="Any language to georgian translator",
    instructions="You translate the user's message to georgian "
)

# -ae
icelandic_agent = Agent(
    name="icelandic_agent",
    handoff_description="Any language to icelandic translator",
    instructions="You translate the user's message to icelandic "
)

# -af
hungarian_agent = Agent(
    name="hungarian_agent",
    handoff_description="Any language to hungarian translator",
    instructions="You translate the user's message to hungarian "
)

# -ag
kazakh_agent = Agent(
    name="kazakh_agent",
    handoff_description="Any language to kazakh translator",
    instructions="You translate the user's message to kazakh "
)

# -ah
latvian_agent = Agent(
    name="latvian_agent",
    handoff_description="Any language to latvian translator",
    instructions="You translate the user's message to latvian "
)

# -ai
dutch_agent = Agent(
    name="dutch_agent",
    handoff_description="Any language to dutch translator",
    instructions="You translate the user's message to dutch "
)

# -aj
malay_agent = Agent(
    name="malay_agent",
    handoff_description="Any language to malay translator",
    instructions="You translate the user's message to malay "
)

# -ak
serbian_agent = Agent(
    name="serbian_agent",
    handoff_description="Any language to serbian translator",
    instructions="You translate the user's message to serbian "
)

# -al
somali_agent = Agent(
    name="somali_agent",
    handoff_description="Any language to somali translator",
    instructions="You translate the user's message to somali "
)

# -am
tagalog_agent = Agent(
    name="tagalog_agent",
    handoff_description="Any language to tagalog translator",
    instructions="You translate the user's message to tagalog "
)

# -an
tamil_agent = Agent(
    name="tamil_agent",
    handoff_description="Any language to tamil translator",
    instructions="You translate the user's message to tamil "
)

# -ao
thai_agent = Agent(
    name="thai_agent",
    handoff_description="Any language to thai translator",
    instructions="You translate the user's message to thai "
)

# -ap
vietnamese_agent = Agent(
    name="vietnamese_agent",
    handoff_description="Any language to vietnamese translator",
    instructions="You translate the user's message to vietnamese "
)

# -aq
welsh_agent = Agent(
    name="welsh_agent",
    handoff_description="Any language to welsh translator",
    instructions="You translate the user's message to welsh "
)

# -ar
merathi_agent = Agent(
    name="merathi_agent",
    handoff_description="Any language to merathi translator",
    instructions="You translate the user's message to merathi "
)

# -as
ukrainian_agent = Agent(
    name="ukrainian_agent",
    handoff_description="Any language to ukrainian translator",
    instructions="You translate the user's message to ukrainian "
)

# -at
yiddish_agent = Agent(
    name="yiddish_agent",
    handoff_description="Any language to yiddish translator",
    instructions="You translate the user's message to yiddish "
)

# -au
indonesian_agent = Agent(
    name="indonesian_agent",
    handoff_description="Any language to indonesian translator",
    instructions="You translate the user's message to indonesian "
)


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[urdu_agent,english_agent,spanish_agent,arabic_agent,german_agent,hindi_agent,russian_agent,
              italian_agent,chinies_agent,japanese_agent,bangali_agent,nepali_agent,punjabi_agent,korean_agent,
              albanian_agent,turkish_agent,pashto_agent,gujarati_agent,french_agent,latin_agent,persian_agent,
              filipino_agent,danish_agent,dari_agent,bosnian_agent,catalan_agent,croatian_agent,finnish_agent,
              georgian_agent,icelandic_agent,hungarian_agent,kazakh_agent,latvian_agent,dutch_agent,malay_agent,
              serbian_agent,somali_agent,tagalog_agent,tamil_agent,thai_agent,vietnamese_agent,welsh_agent,merathi_agent,
              ukrainian_agent,yiddish_agent,indonesian_agent
              ]
)

# Step-5: Start chat

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the English to Arabic, Urdu, hindi, italian, russian, german and spanish translator Agent. How i can help you today?").send()

# Step-6: Runner

@cl.on_message
async def handel_message(message: cl.Message):
   
    
    msg = cl.Message(content="Thinking.......\n\n")
    await msg.send()
    
    history =cl.user_session.get("history")
    history.append({"role":"user", "content":message.content})
    
    result = Runner.run_streamed(
        triage_agent,
        input=history,
        run_config=run_config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        
    history.append({"role":"assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    



