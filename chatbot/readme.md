# Stock chatbot using OpenAI API ([Link](https://stock-chatbot-tool.streamlit.app/))
With this chatbot, you can query the stock price of one or more public companies. You can ask about the performance of 1 or more stocks over specified days and to aid you, a line plot will be shown to observe the trend. Currently, the chatbot is highly dependent on Yahoo Finance API.

Current Features/Tools for the agent:
- Stock closing price inspector with user-specified periods
- Line/Candlestick plot
- Multiple line plot for price comparison
- Stock price performance comparison for >2 tickers and the best-performing stock
- Searching Tool

Use the `.env` file on your local machine, clone the repo, and run it locally if you prefer not to input your API key.  
Steps:
1. Install requirements.txt `pip install -r requirements.txt`
2. Run app `streamlit run app.py`
3. Add more features as you like!
