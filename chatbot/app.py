import yfinance as yf
import streamlit as st

from langchain.agents import initialize_agent
from langchain.agents import AgentType
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from agent_tools import *

# You cannot provide any plot, so do not include it in the final answer.
TEMPLATE = """
You are a helpful chatbot that can answer queries related to Australian stock market. You do not and must not create a plot.
If number of days is not specified, the default number of days is 30.
The plotting tools are provided for you to use. If a plot is requested, it will return the float percentage change increase or decrease, you need to interpret it.
Or if multiple stocks plot is requested, the tool will return the best performing stock and the float percentage change change increase or decrease, you need to interpret it.
Remember to put ".AX" at the end of a stock ticker to request only for Australian stocks. Otherwise do not append ".AX".
Query: {query}
"""
    
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
    # plot_check = st.checkbox('Display plot')
    # # Load OpenAI API key from .env
    # load_dotenv()
    # Tools for our agent to use
    tools = [ddg_tool, 
             StockPriceTool(), StockPctChangeTool(), StockGetBestPerformingTool(), 
             StockGetCandlestickTool(), StockGetHistoricalPlotTool(), StockGetMultiplePlotTool()]

    prompt_template = PromptTemplate(
        input_variables=['query'],
        template=TEMPLATE
    )
    down_flag = False
    try:
        a = yf.Ticker('TSLA').info
        del a
    except Exception as e:
        st.warning("""
            The Yahoo Finance API is currently down.\n
            The chatbot highly depends on the API functionalities, so it won't be able to run without it.
            """)
        print(e)
        down_flag = True
    if not down_flag:
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
            if not api_key:
                st.info("Please input your OpenAI API key to use the chatbot")
            ## GPT Model
            else:
                llm = OpenAI(temperature=0, streaming=True, openai_api_key=api_key)
    
        if llm and api_key:
            # days_ago = st.radio(
            #     'Number of days to look back',
            #     ['7 days', '14 days', '21 days', '30 days', '60 days', '90 days', '180 days', '365 days'],
            #     index=3, horizontal=True)
            open_ai_agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                return_intermediate_steps=True,
                verbose=True
            )
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {'role':'assistant', 'content':"Welcome! how can I help you today?"}
                ]
                
            for msg in st.session_state.messages:
                st.chat_message(msg['role']).write(msg['content']) 
            
            if query := st.chat_input(placeholder="What is the stock price of Commonwealth Bank of Australia?"):
                prompt = prompt_template.format(query=query)
                st.session_state.messages.append({'role':'user','content':query})
                st.chat_message("user").write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner('Generating ...'):
                        # st_callback = StreamlitCallbackHandler(st.container())
                        output = open_ai_agent(prompt)
                        response = output['output']
                        st.session_state.messages.append({'role':'assistant','content':response})
                        st.write(response)
                        # get the message arguments for plotting
                        # steps_dict = output['intermediate_steps'][-1][0].dict()
                        # _args = steps_dict.get('tool_input')
                        # if len(_args.split('|'))>1:
                        #     stocks, days_ago = _args.split('|')
                        # elif len(_args.split(','))>1:
                        #     code, days_ago = _args.split(',')
                        # else:
                        #     code = _args
                        #     days_ago = 30
                        
                        # if set(['plot', 'graph', 'chart', 'historical','compare']).intersection(set(prompt.lower().split())):
                        #     if 'candlestick' in prompt.lower():
                        #         pass
                        #         if int(days_ago)>0:
                        #             st.plotly_chart(get_candlestick_plot(code, int(days_ago)), use_container_width=True)
                        #     else:
                        #         plot_flag = False
                        #         try:
                        #             if int(days_ago)>0:
                        #                 st.plotly_chart(get_hist_price_plot(code, int(days_ago)), use_container_width=True)
                        #                 plot_flag = True

                        #         except Exception as e:
                        #             print('Trying plot for multiple stock tickers')
                                    
                        #         try:
                        #             if not plot_flag:
                        #                 stocks, days_ago = _args.split('|')
                        #                 if int(days_ago)>0:
                        #                     st.plotly_chart(get_multiple_price_plot(eval(stocks), int(days_ago)), use_container_width=True)
                                
                        #         except Exception as e:
                        #             print(e)
