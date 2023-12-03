import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.tools import BaseTool
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun

from typing import Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

def get_stock_price(code: str) -> str:
    """
    Get the closing price of a given stock code.
    
    Args:
        code (str): The stock code for which the closing price is to be retrieved.
        
    Returns:
        str: A string object containing the closing price of the stock rounded to two decimal places and the currency symbol.
    """
    ticker = yf.Ticker(code)
    try:
        currency = ticker.info['currency']
    except:
        print('Currency info is not available')
        currency = 'AUD'
    todays_data = ticker.history(period='1d')
    closing_price = todays_data['Close'].iloc[0]
    formatted_price = "{:.2f} {}".format(closing_price, currency)
    return formatted_price

def get_history_prices(code: str, days_ago: int) -> tuple:
    """
    Get historical data of a given stock code for n days ago.
    
    Args:
        code (str): The stock code for which the closing price is to be retrieved.
        days_ago (int): The number of days to look back.
    
    Returns:
        tuple: A pandas dataframe object and the currency associated.
    """
    ticker = yf.Ticker(code)
    print(ticker.info.keys())
    try:
        currency = ticker.info['currency']
    except:
        print('Currency info is not available')
        currency = 'AUD'
    # get the start_date(today - days_ago) and the end_date(today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    # convert to string
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    # get the data
    hist_data = ticker.history(start=start_date, end=end_date)
    
    return hist_data, currency

def get_price_change_pct(stockticker_days: str) -> float:
    """
    Get the change percentage of a stock price.
    
    Args:
        stockticker_days (str): A string object in a format of comma-separated {ticker, days} ex: (CBA.AX,10)
        
    Returns:
        float: The percentage of change rounded to two decimals.
    """
    code, days_ago = stockticker_days.split(',')
    hist_data, _ = get_history_prices(code, int(days_ago)) # get the history data
    
    old_price = hist_data['Close'].iloc[0] # the first price from (today - days_ago) ~ today
    new_price = hist_data['Close'].iloc[-1] # the last price/current price
    
    pct_change = ((new_price - old_price) / old_price) * 100 # % of change
    
    return round(pct_change, 2)

def get_hist_price_plot(stockticker_days) -> go.Figure:
    """
    Create a line plot of a given stock code over days ago. 
    
    Args:
        code (str): The stock code for which the closing price is to be retrieved.
        days_ago (int): The number of days to look back.
        
    Returns:
        plotly.graph_objects.Figure: The figure object storing the line plot.
    """
    code, days_ago = stockticker_days.split(',')
    hist_data, currency = get_history_prices(code, int(days_ago)) # get history data
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    return get_price_change_pct(stockticker_days)

def get_multiple_price_plot(stocktickers_days) -> go.Figure:
    """
    Get the line plot for multiple stocks to be plotted in one figure.
    
    Args:
        stocks (list): List of stock code for which the closing price is to be retrieved.
        days_ago (int): The number of days to look back.
        
    Returns:
        plotly.graph_objects.Figure: The figure object storing the line plot.
    """
    stocks, days_ago = stocktickers_days.split('|') # input is '[]|int'
    stocks = eval(stocks) # treat string object as a list
    days_ago = int(days_ago) # convert to int
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
        title=f"{' vs '.join(stocks)} stock price comparison over {days_ago} days - now",
        markers=True,
        color_discrete_sequence=['red', 'blue']
    ).update_layout(
        xaxis={'title':'Date'}, 
        yaxis={"title": f"Price ({currency})"}, 
        legend={"title":"Code"}
    )
    fig.update_layout(hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)
    
    return get_best_performing_stock(stocktickers_days)

def get_candlestick_plot(stockticker_days) -> go.Figure:
    """
    Get the candlestick plot of {code} over {days_ago}
    
    Args:
        code (str): The stock code for creating the candlestick plot.
        days_ago (int): The number of days to look back.
        
    Returns:
        plotly.graph_objects.Figure: The figure object storing the candlestick figure.
    """
    code, days_ago = stockticker_days.split(',')
    hist_data, currency = get_history_prices(code, int(days_ago)) # get history data
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
    st.plotly_chart(fig, use_container_width=True)
    return get_price_change_pct(stockticker_days)

def get_best_performing_stock(stocktickers_days):
    """
    Get the best performing stock from the list of stock codes over days ago. Input is a string to be used by the agent.
    
    Args:
        stocktickers_days (str): A string object in a comma-separated format of {list, int} ex: "[CBA.AX,BHP.AX,MQG.AX]|10"
    
    Returns:
        tuple: A tuple consisting of the stock ticker and percentage of change.
    """
    stocks, days_ago = stocktickers_days.split('|') # input is '[]|int'
    stocks = eval(stocks) # treat string object as a list
    days_ago = int(days_ago) # convert to int
    # create a starting point/benchmark
    best_stock = stocks[0]
    best_performance = get_price_change_pct(f"{stocks[0]},{days_ago}")
    # iterate over the rest of {stocks}
    for code in stocks[1:]:
        try:
            current_performance = get_price_change_pct(f"{code},{days_ago}") # get % of change
            # compare with best performer
            if current_performance>best_performance:
                best_stock = code
                best_performance = current_performance
                
        except Exception as e:
            print(f"Could not calculate performance for {code}: {e}")
            
    return best_stock, best_performance

### Tools for the agent
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
    description = """
    Use this only when you need to find out today's price of a stock and not price over a period of time. 
    You should input the stock ticker used on the yfinance API. 
    """

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
    description = """
    Use this only when you need to find out the percentage change or trend or stock price over a period of time for ONLY one company. 
    You should input the stock ticker used on the yfinance API and also input the number of days to check the change over. 
    The input to this tool should be a comma separated list with length of 2 consisting of string representing the stock ticker and number representing days to look back. 
    For example, `MSFT,30` would be the input if you wanted to find out Microsoft stock price over 30 days ago.
    """
    
    def _run(self, stockticker_days: str):
        price_change_response = get_price_change_pct(stockticker_days)
        return price_change_response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")
    
    args_schema: Optional[Type[BaseModel]] = StockPctChangeCheckInput
    
class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker. For performance comparison percentage check"""

    input_string: str = Field(..., description='A string of a comma separated list of strings and a number')
    
class StockGetBestPerformingTool(BaseTool):
    name = 'get_best_performing_stock'
    description = """
    Use this only when you need to compare the performance/stock price/trend of two or more companies over a period days. 
    You should input a python list of stock tickers used on the yfinance API and also input the number of days to check the change over. 
    The input to this tool should be a comma separated list consisting of strings representing the stock tickers without space and a number representing days to look back. 
    The list and the number should be separated by |. For example, `['MSFT','AAPL','GOOGL']|30` would be the input if you wanted to compare Microsoft, Apple, Google stock price over 30 days ago. 
    """
    
    def _run(self, stockticker_days: str):
        response = get_best_performing_stock(stockticker_days)

        return response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput
    
class StockHistoricalPlotInput(BaseModel):
    """Input for Stock ticker. For plotting historical prices of a stock."""

    input_string: str = Field(..., description='a comma separated string with length of 2 consisting of string and number')  
    
class StockGetHistoricalPlotTool(BaseTool):
    name = 'get_historical_prices_plot'
    description = """
    Use this only when you need to get the historical prices of a stock over a period of time. 
    This returns the percentage change over a period of time.
    You should input the stock ticker used on the yfinance API and also input the number of days to check the change over. 
    The input to this tool should be a string representing the stock ticker, a comma, and a number representing days to look back. 
    For example, `MSFT,30` would be the input if you wanted to find out Microsoft stock price over 30 days ago.
    """
    
    def _run(self, stockticker_days: str):
        response = get_hist_price_plot(stockticker_days)

        return response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockHistoricalPlotInput
    
class StockGetMultiplePlotInput(BaseModel):
    """Input for Stock ticker. For plotting multiple stocks."""

    input_string: str = Field(..., description='a comma separated string with length of 2 consisting of string and number')
    
class StockGetMultiplePlotTool(BaseTool):
    name = 'get_multiple_stocks_price_plot'
    description = """
    Use this only when you need to get the multiple plots of a stock over a period of time for comparison. 
    This returns the best performing stock, and the percentage performace.
    You should input a python list of stock tickers used on the yfinance API and also input the number of days to check the change over. 
    The input to this tool should be a comma separated list consisting of strings representing the stock tickers without space and a number representing days to look back. 
    The list and the number should be separated by |. For example, `['MSFT','AAPL','GOOGL']|30` would be the input if you wanted to compare Microsoft, Apple, Google stock price over 30 days ago. 
    """
    
    def _run(self, stocktickers_days: str):
        response = get_multiple_price_plot(stocktickers_days)

        return response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockGetMultiplePlotInput
    
class StockGetCandlestickInput(BaseModel):
    """Input for Stock ticker. For plotting candlestick"""

    input_string: str = Field(..., description='a comma separated string with length of 2 consisting of string and number')
    
class StockGetCandlestickTool(BaseTool):
    name = 'get_candlestick_plot'
    description = """
    Use this only when you need to get the candlestick plot of a stock over a period of time.
    Use get_historical_prices_plot tool if candlestick is not specified.
    This returns the percentage change over a period of time.
    You should input the stock ticker used on the yfinance API and also input the number of days to check the change over. 
    The input to this tool should be a string representing the stock ticker, a comma, and a number representing days to look back. 
    For example, `MSFT,30` would be the input if you wanted to find out Microsoft stock price over 30 days ago.
    """
    
    def _run(self, stockticker_days: str):
        response = get_candlestick_plot(stockticker_days)

        return response
    
    def _arun(self, stockticker_days: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput