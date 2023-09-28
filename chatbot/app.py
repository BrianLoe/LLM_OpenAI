import yfinance as yf
import os
import streamlit as st
# import matplotlib.pyplot as plt
import json
# from matplotlib.dates import MonthLocator, DateFormatter
from typing import Optional, Type
from langchain.tools import BaseTool, format_tool_to_openai_function
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime, timedelta
# from bokeh.plotting import figure
# from bokeh.models import DatetimeTickFormatter
import plotly.express as px

def get_stock_price(code):
    ticker = yf.Ticker(code)
    currency = ticker.info['currency']
    todays_data = ticker.history(period='1d')
    return "{} {}".format(round(todays_data['Close'].iloc[0], 2), currency)

def get_history_prices(code, days_ago):
    ticker = yf.Ticker(code)
    currency = ticker.info['currency']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    hist_data = ticker.history(start=start_date, end=end_date)
    
    return hist_data, currency

def get_price_change_pct(code, days_ago):    
    hist_data, _ = get_history_prices(code, days_ago)
    
    old_price = hist_data['Close'].iloc[0]
    new_price = hist_data['Close'].iloc[-1]
    
    pct_change = ((new_price - old_price) / old_price) * 100
    
    return round(pct_change, 2)

def get_hist_price_plot(code, days_ago):
    hist_data, currency = get_history_prices(code, days_ago)
    hist_data = hist_data['Close']
    fig = px.line(
        x=hist_data.index,
        y=hist_data.values,
        title=f"{code} stock price over {days_ago} days - now",
        labels={
            'x': 'Date',
            'y': f'Price ({currency})'
        }
    )
    
    return fig

class StockPriceCheckInput(BaseModel):
    """Input for stock price check."""

    stockticker: str = Field(..., description="Ticker code for stock or index")

class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        # print("i'm running")
        price_response = get_stock_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput
    
class StockPctChangeCheckInput(BaseModel):
    """Input for stock ticker check for percentage check"""
    
    stockticker: str = Field(..., description='Ticker symbol for stock or index')
    days_ago: int = Field(..., description='Int number of days to look back')
    
class StockPctChangeTool(BaseTool):
    name = 'get_price_change_percent'
    description = "Useful when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"
    
    def _run(self, stockticker: str, days_ago: int):
        price_change_response = get_price_change_pct(stockticker, days_ago)
        return price_change_response
    
    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")
    
    args_schema: Optional[Type[BaseModel]] = StockPctChangeCheckInput
    
if __name__=='__main__':
    # Load OpenAI API key from .env
    load_dotenv()
    # Search tool using duckduckgo
    search = DuckDuckGoSearchRun()
    ddg_tool = Tool(
        name='search',
        func=search.run,
        description='useful for when you need to answer questions about current events. You should ask targeted questions'
    )
    # Web UI Layout
    st.set_page_config(
        page_title='Australia Stock Price AI Tool',
        page_icon='📈',
        layout='centered'
    )
    st.title("OpenAI Stock Chatbot")
    st.info("You can ask the bot about: today's stock price, australian companies who are listed on ASX, etc.")
    # plot_check = st.checkbox('Display plot')
    # GPT Model
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo-0613")
    # Tools for our agent to use
    tools = [ddg_tool, StockPriceTool(), StockPctChangeTool()]
    
    arg_tool = [StockPctChangeTool()]
    functions = [format_tool_to_openai_function(t) for t in arg_tool]
    
    open_ai_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {'role':'assistant', 'content':"G'day, how can I help you today?"}
        ]
        
    for msg in st.session_state.messages:
      st.chat_message(msg['role']).write(msg['content'])  
    if prompt := st.chat_input(placeholder="What is the stock price of Commonwealth Bank of Australia?"):
        st.session_state.messages.append({'role':'user','content':prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            st.spinner('Generating ...')
            st_callback = StreamlitCallbackHandler(st.container())
            response = open_ai_agent.run(prompt, callbacks=[st_callback])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write(response)
            # get the message arguments for plotting
            ai_message = llm.predict_messages([HumanMessage(content=prompt)], functions=functions)
            _args = json.loads(ai_message.additional_kwargs['function_call'].get('arguments'))
            stock_code = _args.get('stockticker')
            days = _args.get('days_ago',0)
            if days>0:
                st.plotly_chart(get_hist_price_plot(stock_code, days), use_container_width=True)