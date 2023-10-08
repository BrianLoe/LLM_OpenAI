import yfinance as yf
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from langchain.tools import BaseTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler

from typing import Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime, timedelta

TEMPLATE = """
If a plot is requested, don't try to create a plot, just answer the rest of the question using the tools provided.
Query: {query}
"""

def get_stock_price(code):
    """
    Get the closing price of {code}.
    Returns a string object with format of price rounded to two decimals + currency
    """
    ticker = yf.Ticker(code)
    currency = ticker.info['currency'] # get the currency info
    todays_data = ticker.history(period='1d')
    return "{} {}".format(round(todays_data['Close'].iloc[0], 2), currency)

def get_history_prices(code, days_ago):
    """
    Get historical data of {code} based on {days_ago}
    Returns a pandas dataframe object and the currency associated.
    """
    ticker = yf.Ticker(code)
    currency = ticker.info['currency']
    # get the start_date(today - days_ago) and the end_date(today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    # convert to string
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    # get the data
    hist_data = ticker.history(start=start_date, end=end_date)
    
    return hist_data, currency

def get_price_change_pct(stockticker_days):
    """
    Get the change percentage of a stock price. Input is a string object of comma separated list consisting 'stockticker,days'.
    Returns the percentage in float rounded to two decimals.
    """
    code, days_ago = stockticker_days.split(',')
    hist_data, _ = get_history_prices(code, int(days_ago)) # get the history data
    
    old_price = hist_data['Close'].iloc[0] # the first price from (today - days_ago) ~ today
    new_price = hist_data['Close'].iloc[-1] # the last price/current price
    
    pct_change = ((new_price - old_price) / old_price) * 100 # % of change
    
    return round(pct_change, 2)

def get_hist_price_plot(code, days_ago):
    """
    Get the line plot of {code} over {days_ago}. 
    Returns the figure object of a {code} trend line plot over {days_ago}
    """
    hist_data, currency = get_history_prices(code, days_ago) # get history data
    hist_data = hist_data['Close'] # take the closing price
    # make a line plot
    fig = px.line(
        x=hist_data.index,
        y=hist_data.values,
        title=f"{code} stock price over {days_ago} days - now",
        labels={
            'x': 'Date',
            'y': f'Price ({currency})'
        }
    ).update_traces(line_color='gray')
    # make annotation for highest/lowest price
    max_idx, max_val = hist_data.idxmax(), hist_data[hist_data.idxmax()]
    min_idx, min_val = hist_data.idxmin(), hist_data[hist_data.idxmin()]
    # add annotation
    fig.add_trace(go.Scatter(
        x=[max_idx],
        y=[max_val],
        mode='text+markers',
        text=['Highest'],
        textposition="middle left",
        showlegend=False,
        marker=dict(
            color='crimson',
            symbol='triangle-up',
            size=13
        ),
        textfont=dict(
            size=14,
            color="crimson"
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=[min_idx],
        y=[min_val],
        mode='text+markers',
        text=['Lowest'],
        textposition="middle left",
        showlegend=False,
        marker=dict(
            color='blue',
            symbol='triangle-down',
            size=13
        ),
        textfont=dict(
            size=14,
            color="blue"
        )
    ))
    
    return fig

def get_multiple_price_plot(stocks, days_ago):
    """
    Get the line plot for multiple {stocks} to be plotted in one figure.
    Returns the figure object that shows the comparison of {stocks} over {days_ago}
    """
    df_dict = dict.fromkeys(stocks,[]) # make a dictionary from {stocks} as keys and empty list as value
    # iterate over the {stocks}
    for code in stocks:
        hist_data, currency = get_history_prices(code, days_ago) # get the history data
        df_dict[code] = hist_data['Close'].values # take the closing price and store it into dictionary
    # make a dataframe object
    df = pd.DataFrame(df_dict, index=hist_data.index)
    # make a line plot
    fig = px.line(
        df,
        title=f"{' vs '.join(stocks)} stock price comparison over {days_ago} days - now"
    ).update_layout(
        xaxis={'title':'Date'}, 
        yaxis={"title": f"Price ({currency})"}, 
        legend={"title":"Code"}
    )
    
    return fig

def get_candlestick_plot(code, days_ago):
    """
    Get the candlestick plot of {code} over {days_ago}
    Returns the figure object that shows the candlestick figure.
    """
    hist_data, currency = get_history_prices(code, days_ago) # get history data
    # string format the start_date and end_date for making the title
    start_date = hist_data.index.min().date().strftime('(%d %b %Y -')
    end_date = hist_data.index.max().date().strftime(' %d %b %Y)')
    # make the candlestick plot
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                
            )
        ]
    ).update_layout(
        xaxis={'title':'Date'}, 
        yaxis={"title": f"Price ({currency})"}, 
        title=f'{code} {start_date + end_date}'
    )
    
    return fig

def get_best_performing_stock(stocktickers_days):
    """
    Get the best performing stock from {stocktickers} over {days}. Input is a string to be used by agent.
    Returns a tuple consisting of the stock ticker, percentage of change
    """
    stocks, days_ago = stocktickers_days.split(', ') # input is '[], int'
    stocks = eval(stocks) # treat string object as a list
    days_ago = int(days_ago) # convert to int
    # create a starting point/benchmark
    best_stock = stocks[0]
    best_performance = get_price_change_pct(str(stocks[0])+','+str(days_ago))
    # iterate over the rest of {stocks}
    for code in stocks[1:]:
        try:
            perf = get_price_change_pct(str(code)+','+str(days_ago)) # get % of change
            # compare with current best performer
            if perf>best_performance:
                best_stock = code
                best_performance = perf
                
        except Exception as e:
            print(f"Could not calculate performance for {code}: {e}")
            
    return best_stock, best_performance

# Search tool using duckduckgo
search = DuckDuckGoSearchRun()
ddg_tool = Tool(
        name='search',
        func=search.run,
        description='Useful for when you need to answer questions about current events/news and questions related that cannot be answered using other tools. Only search answers related to the company stock. You should ask targeted questions'
    )

class StockPriceCheckInput(BaseModel):
    """Input for stock price check."""

    stockticker: str = Field(..., description="Ticker code for stock or index")

class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Use this only when you need to find out today's price of a stock and not price over a period of time. You should input the stock ticker used on the yfinance API. You don't need to do anything if plot is requested."

    def _run(self, stockticker: str):
        # print("i'm running")
        price_response = get_stock_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput
    
class StockPctChangeCheckInput(BaseModel):
    """Input for stock ticker for percentage or trend check"""
    
    input_string: str = Field(..., description='a comma separated list with length of 2 consisting of string and number')
    
class StockPctChangeTool(BaseTool):
    name = 'get_price_change_percent'
    description = "Use this only when you need to find out the percentage change or trend or stock price over a period of time for ONLY one company. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over. The input to this tool should be a comma separated list with length of 2 consisting of string representing the stock ticker and number representing days to look back. For example, `MSFT,30` would be the input if you wanted to find out Microsoft stock price over 30 days ago. You don't need to do anything if plot is requested."
    
    def _run(self, stockticker_days: str):
        price_change_response = get_price_change_pct(stockticker_days)
        return price_change_response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")
    
    args_schema: Optional[Type[BaseModel]] = StockPctChangeCheckInput
    
class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker. For performance comparison percentage check"""

    input_string: str = Field(..., description='a comma separated list with length of 2 consisting of list of strings and a number')
    
class StockGetBestPerformingTool(BaseTool):
    name = 'get_best_performing_stock'
    description = "Use this only when you need to compare the performance/stock price/trend of two or more companies over a period days. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over. The input to this tool should be a comma separated list with length of 2 consisting of list of strings representing the stock tickers (without space) and a number representing days to look back. For example, `['MSFT','AAPL','GOOGL'], 30` would be the input if you wanted to compare/find out Microsoft, Apple, Google stock price over 30 days ago. You don't need to do anything if plot is requested."
    
    def _run(self, stockticker_days: str):
        response = get_best_performing_stock(stockticker_days)

        return response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput
    
if __name__=='__main__':
    # Streamlit UI Layout
    st.set_page_config(
        page_title='Stock Price Inspector AI Tool',
        page_icon='📈',
        # layout='centered'
    )
    st.title("OpenAI Stock Chatbot")
    st.markdown("""
                You can ask the bot about: 
                price of a stock, trend of a stock over n days, performance comparison of two or more stocks, 
                and search about latest news of a public listed company. Explicitly specify if you 
                need a trend/line plot or candlestick plot. Always specify the stock and days you want to look 
                for when requesting for a plot.
                """)
    # plot_check = st.checkbox('Display plot')
    ## GPT Model
    st.markdown("""
                You need to input your OPENAI_API_key to use OpenAI GPT Model.
                """)
    api_key = st.text_input('Input your OPENAI_API_key', type='password')
    # # Load OpenAI API key from .env
    # load_dotenv()
    llm = OpenAI(temperature=0, streaming=True, openai_api_key=api_key)
    # Tools for our agent to use
    tools = [ddg_tool, StockPriceTool(), StockPctChangeTool(), StockGetBestPerformingTool()]
    
    open_ai_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True,
        verbose=True
    )
    
    prompt_template = PromptTemplate(
        input_variables=['query'],
        template=TEMPLATE
    )
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {'role':'assistant', 'content':"G'day, how can I help you today?"}
        ]
        
    for msg in st.session_state.messages:
      st.chat_message(msg['role']).write(msg['content']) 
      
    if query := st.chat_input(placeholder="What is the stock price of Commonwealth Bank of Australia?"):
        prompt = prompt_template.format(query=query)
        st.session_state.messages.append({'role':'user','content':prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner('Generating ...'):
                # st_callback = StreamlitCallbackHandler(st.container())
                output = open_ai_agent(prompt)
                response = output['output']
                st.session_state.messages.append({'role':'assistant','content':response})
                st.write(response)
                # get the message arguments for plotting
                steps_dict = output['intermediate_steps'][0][0].dict()
                _args = steps_dict.get('tool_input')
                
                if set(['plot', 'graph', 'chart']).intersection(set(prompt.lower().split())):
                    if 'candlestick' in prompt.lower():
                        code, days_ago = _args.split(',')
                        if int(days_ago)>0:
                            st.plotly_chart(get_candlestick_plot(code, int(days_ago)), use_container_width=True)
                    else:
                        plot_flag = False
                        try:
                            code, days_ago = _args.split(',')
                            if int(days_ago)>0:
                                st.plotly_chart(get_hist_price_plot(code, int(days_ago)), use_container_width=True)
                                plot_flag = True

                        except Exception as e:
                            print('Trying plot for multiple stock tickers')
                            
                        try:
                            if not plot_flag:
                                stocks, days_ago = _args.split(', ')
                                if int(days_ago)>0:
                                    st.plotly_chart(get_multiple_price_plot(eval(stocks), int(days_ago)), use_container_width=True)
                        
                        except Exception as e:
                            print(e)