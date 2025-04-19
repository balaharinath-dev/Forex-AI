import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Rest of your imports and code below
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
import requests
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Forex Analysis Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tool Definitions ---
class ExchangeRateToolInput(BaseModel):
    """Input schema for Exchange Rate Tool."""
    currency_pair: str = Field(..., description="Currency pair in format BASE,TARGET (e.g., USD,INR)")

class ExchangeRateTool(BaseTool):
    name: str = "get_exchange_rate"
    description: str = "Get the current exchange rate between two currencies. Input should be a comma-separated string with base_currency and target_currency, e.g., 'USD,INR'"
    args_schema: Type[BaseModel] = ExchangeRateToolInput

    def _run(self, currency_pair: str) -> Dict[str, Any]:
        """Get the current exchange rate between two currencies."""
        base_currency, target_currency = currency_pair.split(',')
        api_key = st.secrets("EXCHANGE_RATE_API_KEY", "")
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "base_currency": base_currency,
                    "target_currency": target_currency,
                    "rate": data["conversion_rate"],
                    "last_updated": data["time_last_update_utc"],
                    "next_update": data["time_next_update_utc"],
                    "status": "success"
                }
            return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class ForexNewsToolInput(BaseModel):
    """Input schema for Forex News Tool."""
    currency_pair: str = Field(..., description="Currency pair in format BASE/TARGET (e.g., USD/INR)")

class ForexNewsTool(BaseTool):
    name: str = "get_forex_news"
    description: str = "Get the latest news related to a specific currency pair. Input should be a currency pair in format 'BASE/TARGET', e.g., 'USD/INR'"
    args_schema: Type[BaseModel] = ForexNewsToolInput

    def _run(self, currency_pair: str) -> List[Dict[str, Any]]:
        """Get the latest news related to a specific currency pair."""
        base, target = currency_pair.split('/')
        api_key = st.secrets("NEWS_API_KEY", "")
        url = "https://newsapi.org/v2/everything"
        query = f"forex {base} {target} exchange rate"
        
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 5,
            "apiKey": api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "title": article.get("title"),
                        "source": article.get("source", {}).get("name"),
                        "published_at": article.get("publishedAt"),
                        "url": article.get("url"),
                        "description": article.get("description")
                    }
                    for article in data.get("articles", [])
                ]
            return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class TrendAnalysisToolInput(BaseModel):
    """Input schema for Trend Analysis Tool."""
    currency_pair: str = Field(..., description="Currency pair in format BASE/TARGET (e.g., USD/INR)")

class TrendAnalysisTool(BaseTool):
    name: str = "analyze_forex_trends"
    description: str = "Analyze trends for a currency pair. Input should be a currency pair in format 'BASE/TARGET', e.g., 'USD/INR'"
    args_schema: Type[BaseModel] = TrendAnalysisToolInput

    def _run(self, currency_pair: str) -> Dict[str, Any]:
        """Analyze trends for a currency pair."""
        import random
        base, target = currency_pair.split('/')
        trend_directions = ["upward", "downward", "sideways"]
        sentiment_options = ["bullish", "bearish", "neutral"]
        
        return {
            "base_currency": base,
            "target_currency": target,
            "current_trend": random.choice(trend_directions),
            "trend_strength": random.randint(3, 8),
            "public_sentiment": random.choice(sentiment_options),
            "volatility": random.uniform(0.1, 5.0),
            "trading_volume": f"{random.randint(50, 500)}M",
            "note": "This is simplified trend data for demonstration purposes."
        }

# --- Forex Analysis System ---
class ForexAnalysisSystem:
    """A complete system for analyzing forex currency pairs using CrewAI."""
    
    def __init__(self):
        """Initialize the system with required models and tools."""
        self.gemini_api_key = st.secrets("GEMINI_API_KEY", "")
        
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            top_p=0.8,
        )
        
        # Initialize tools
        self.exchange_rate_tool = ExchangeRateTool()
        self.forex_news_tool = ForexNewsTool()
        self.trend_analysis_tool = TrendAnalysisTool()
        
        # Initialize agents and crew
        self._setup_agents()
        self._setup_crew()
    
    def _setup_agents(self):
        """Set up the agents for the forex analysis crew."""
        # Data Collection Agent
        self.data_collector = Agent(
            role="Forex Data Collector",
            goal="Collect accurate and up-to-date forex data and news",
            backstory="You are an expert in gathering financial data from various sources. Your specialty is finding the most reliable and current information on currency exchange rates and related news.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.exchange_rate_tool, self.forex_news_tool]
        )
        
        # Analysis Agent
        self.analyst = Agent(
            role="Forex Market Analyst",
            goal="Analyze forex data to identify trends and provide insights",
            backstory="You are a seasoned forex market analyst with decades of experience interpreting currency movements and their implications. You can spot patterns and explain them in ways both experts and novices can understand.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self.trend_analysis_tool]
        )
        
        # Prediction Agent
        self.predictor = Agent(
            role="Forex Prediction Specialist",
            goal="Provide reasonable predictions based on current data and trends",
            backstory="You specialize in forecasting currency movements based on technical analysis, news impact, and market sentiment. You're known for providing balanced and realistic predictions rather than sensationalist claims.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Report Generator
        self.reporter = Agent(
            role="Forex Report Generator",
            goal="Create clear, concise reports that are accessible to both experts and laypeople",
            backstory="You excel at taking complex financial information and translating it into clear, actionable insights. Your reports are known for being straightforward yet comprehensive, with bullet points for clarity.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _setup_crew(self):
        """Set up the CrewAI crew for forex analysis."""
        self.crew = Crew(
            agents=[
                self.data_collector,
                self.analyst,
                self.predictor,
                self.reporter
            ],
            tasks=[],  # We'll add tasks dynamically based on the currency pair
            verbose=True,
            process=Process.sequential
        )
    
    def _create_tasks(self, currency_pair: str):
        """Create tasks for analyzing a specific currency pair."""
        base, target = currency_pair.split('/')
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Task 1: Collect exchange rate data
        collect_rate_task = Task(
            description=f"Collect the current exchange rate for {currency_pair} as of {current_date}",
            expected_output=f"The latest exchange rate data for {currency_pair} including rate and timestamp",
            agent=self.data_collector,
            tools=[self.exchange_rate_tool],
            context=[]
        )
        
        # Task 2: Collect news data
        collect_news_task = Task(
            description=f"Find the latest news articles related to {currency_pair} as of {current_date}",
            expected_output=f"A list of 5 recent news articles that might impact the {currency_pair} exchange rate",
            agent=self.data_collector,
            tools=[self.forex_news_tool],
            context=[]
        )
        
        # Task 3: Analyze trends
        analyze_task = Task(
            description=f"Analyze the current trends for {currency_pair} based on collected data as of {current_date}",
            expected_output=f"Analysis of recent movements and public sentiment regarding {currency_pair}",
            agent=self.analyst,
            tools=[self.trend_analysis_tool],
            context=[collect_rate_task, collect_news_task]
        )
        
        # Task 4: Make predictions
        predict_task = Task(
            description=f"Make short-term and long-term predictions for {currency_pair} based on analysis as of {current_date}",
            expected_output=f"Reasonable short-term (1 week) and long-term (1 month) predictions for {currency_pair} movement",
            agent=self.predictor,
            context=[analyze_task]
        )
        
        # Task 5: Generate report
        report_task = Task(
            description=f"""Create a comprehensive but easy-to-understand report on {currency_pair} as of {current_date} that includes:
            1. Current exchange rate with time of update
            2. Brief summary of recent news (bullet points)
            3. Analysis of current trends and public sentiment
            4. Short-term (1 week) and long-term (1 month) predictions""",
            expected_output=f"A well-formatted report on {currency_pair} with bullet points for clarity",
            agent=self.reporter,
            context=[predict_task]
        )
        
        return [collect_rate_task, collect_news_task, analyze_task, predict_task, report_task]
    
    def analyze_currency_pair(self, currency_pair: str) -> str:
        """Analyze a currency pair and generate a comprehensive report."""
        if '/' not in currency_pair:
            return "Error: Currency pair should be in format 'XXX/YYY' (e.g., 'USD/INR')"
        
        tasks = self._create_tasks(currency_pair)
        self.crew.tasks = tasks
        result = self.crew.kickoff()
        
        # Ensure the result is a string for download
        if not isinstance(result, str):
            result = str(result)
        
        return result

# --- Streamlit App ---
def main():
    """Main function for the Streamlit app."""
    st.title("💹 Forex Currency Pair Analysis")
    st.markdown("""
    This app provides real-time analysis of forex currency pairs, including:
    - Current exchange rates
    - Latest relevant news
    - Public sentiment and trends
    - Future predictions and analysis
    """)
    
    with st.sidebar:
        st.header("📊 About This App")
        st.markdown("""
        ### How It Works
        This app uses a multi-agent AI system powered by CrewAI to analyze currency pairs in depth.
        """)
        st.markdown(f"**Current Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    # Currency selection
    major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    emerging_currencies = ["INR", "CNY", "BRL", "ZAR", "MXN", "RUB", "TRY", "SGD"]
    all_currencies = sorted(list(set(major_currencies + emerging_currencies)))
    
    col1, col2 = st.columns(2)
    with col1:
        base_currency = st.selectbox("Base Currency", options=all_currencies, index=all_currencies.index("USD"))
    with col2:
        target_options = [c for c in all_currencies if c != base_currency]
        default_target = "INR" if "INR" in target_options else target_options[0]
        target_currency = st.selectbox("Target Currency", options=target_options, index=target_options.index(default_target))
    
    currency_pair = f"{base_currency}/{target_currency}"
    
    # Quick converter
    st.subheader("Quick Converter")
    amount = st.number_input(f"Amount in {base_currency}", min_value=0.01, value=1.0, step=0.01)
    
    # Check API keys
    missing_keys = []
    if not st.secrets("EXCHANGE_RATE_API_KEY"):
        missing_keys.append("Exchange Rate API Key")
    if not st.secrets("NEWS_API_KEY"):
        missing_keys.append("News API Key")
    if not st.secrets("GEMINI_API_KEY"):
        missing_keys.append("Gemini API Key")
    
    if missing_keys:
        st.warning(f"Missing API keys: {', '.join(missing_keys)}. Please add them in your .env file.")
    
    if st.button("🔍 Analyze Currency Pair", type="primary", disabled=bool(missing_keys)):
        progress_container = st.empty()
        result_container = st.empty()
        
        with progress_container.container():
            st.subheader("📊 Analysis in Progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = ["Initializing Agents", "Collecting Data", "Analyzing Trends", 
                    "Generating Predictions", "Creating Report"]
            
            for i, step in enumerate(steps):
                progress_bar.progress((i + 1) * 20)
                status_text.text(f"Step {i+1}/{len(steps)}: {step}")
                time.sleep(0.5)
            
            try:
                status_text.text("Running analysis...")
                forex_system = ForexAnalysisSystem()
                result = forex_system.analyze_currency_pair(currency_pair)
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return
        
        progress_container.empty()
        
        with result_container.container():
            st.markdown(f"## Analysis Results: {currency_pair}")
            st.markdown(f"**Report Generated on:** {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
            
            # Single tab for the full report
            st.markdown("### Complete Forex Analysis Report")
            st.markdown(result)
            
            # Download button
            st.download_button(
                label="📥 Download Full Report",
                data=result,
                file_name=f"forex_analysis_{currency_pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()