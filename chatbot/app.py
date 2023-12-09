import yfinance as yf
import streamlit as st
import httpx
from langchain.agents import initialize_agent, ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# from langchain.callbacks import StreamlitCallbackHandler
# from dotenv import load_dotenv
from agent_tools import *

# You cannot provide any plot, so do not include it in the final answer.
PREFIX = """
You are a helpful chatbot that can answer queries related to Australian stock market.  Stock ticker can be not available, and that is okay.
If number of days is not specified, the default number of days is 30.
The plotting tools are provided for you to use. If a plot is requested, you need to interpret the float percentage change.
Remember to append ".AX" for Australian stock tickers. If you don't know the ticker code, you need to search it using the tool.
Don't try to give an answer if you don't know the answer.
"""
SUFFIX = """ Begin!  
{chat_history}
Query: {query}
{agent_scratchpad}
"""

def get_tools():
    """Tools for our agent to use"""
    tools = [ddg_tool, 
             StockPriceTool(), StockPctChangeTool(), StockGetBestPerformingTool(), 
             StockGetCandlestickTool(), StockGetHistoricalPlotTool(), StockGetMultiplePlotTool()]
    return tools

def initialize_agent(llm_model, tools):
    """Initialize the agent with prompt and the chain"""
    prompt = ZeroShotAgent.create_prompt(tools, prefix=PREFIX, suffix=SUFFIX, input_variables=['query', 'chat_history', 'agent_scratchpad'])
    llm_chain = LLMChain(llm=llm_model, prompt=prompt)
    open_ai_agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    
    return open_ai_agent

def check_yf_api():
    """Check if the Yahoo Finance API is up"""
    try:
        a = yf.Ticker('TSLA').info
        return True, None
    except Exception as e:
        return False, str(e)
    
if __name__=='__main__':
    # Streamlit UI Layout
    st.set_page_config(
        page_title='Stock Price Analyzer GPT Tool',
        page_icon='📈',
        # layout='centered'
    )
    with st.container():
        st.title("Stock Chatbot using OpenAI")
        st.markdown("""
                    You can ask the bot about: 
                    - Price of a stock  
                    - Trend of a stock over n days  
                    - Performance comparison of two or more stocks  
                    - Search about latest news of a public listed company  
                    
                    Explicitly specify if you need a trend/line plot or candlestick plot. It is recommended to specify the stock and days you want to query for.
                    Otherwise you can set a default number of days to look back.  
                    
                    **The chatbot is designed to be used for querying Australian stock markets.**
                    """)
    # llm state initialization
    if "llm_initialization_state_flag" not in st.session_state:
        st.session_state.llm_initialization_state_flag = False
        
    tools = get_tools()
    memory = ConversationBufferWindowMemory(memory_key='chat_history',k=2)
    # check for YF API dependency
    up_flag, msg = check_yf_api()
    if not up_flag:
        st.warning("""
            The Yahoo Finance API is currently down.\n
            The chatbot highly depends on the API functionalities, so it won't be able to run without it.
            """)
        print(msg)
    else:
        # check for OpenAI API key
        if 'api_key' not in st.session_state:
            st.info("""
                    You need to input your OpenAI API key to use the chatbot. This chatbot utilizes OpenAI GPT model.
                    """)
        with st.sidebar:
            if 'api_key' not in st.session_state:
                st.session_state.api_key = ''
            def submit():
                st.session_state.api_key = st.session_state.widget
                st.session_state.widget = ''
            st.text_input('Input your OpenAI API key', type='password', key='widget', on_change=submit)
            api_key = st.session_state.api_key
        llm = None
        # initialize llm model
        if api_key:
            # GPT Model (text-davinci-003)
            llm = OpenAI(temperature=0, streaming=True, openai_api_key=api_key)
        else:
            st.info("Please input your OpenAI API key to use the chatbot")
    
        if llm:
            # initialize memory
            if "buffer_memory" not in st.session_state:
                st.session_state["buffer_memory"] = memory   
                
            open_ai_agent = initialize_agent(llm, tools)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=open_ai_agent, tools=tools, verbose=True, memory=st.session_state["buffer_memory"]
            )
            st.session_state["llm_initialization_state_flag"] = True
        # initialize the chatbot agent   
        if st.session_state["llm_initialization_state_flag"]:   
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {'role':'assistant', 'content':"Welcome! how can I help you today?"}
                ]
                
            for msg in st.session_state.messages:
                st.chat_message(msg['role']).write(msg['content']) 
            
            if query := st.chat_input(placeholder="What is the stock price of Commonwealth Bank of Australia?"):
                # prompt = prompt_template.format(query=query)
                st.session_state.messages.append({'role':'user','content':query})
                st.chat_message("user").write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner('Generating ...'):
                        # st_callback = StreamlitCallbackHandler(st.container())
                        try:
                            output = agent_chain.run(query=query)
                        except Exception as e:
                            msg = ''
                            if isinstance(e,httpx.HTTPError):
                                msg = "The DuckDuckGo search tool is currently down. "
                            print(e)
                            output = f"It seems that I wasn't able to answer your query due to an error in my end. {msg}Please try again or open the debugger."
                        # response = output['output']
                        st.session_state.messages.append({'role':'assistant','content':output})
                        st.write(output)
