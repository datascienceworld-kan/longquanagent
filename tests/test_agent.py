import pytest
from langchain_together import ChatTogether
from longquanagent.agent.agent import Agent
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def setup_llm():
    llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    return llm


def test_agent_sum(setup_llm):
    from longquanagent.register.tool import function_tool
    from typing import List

    @function_tool
    def sum_of_series(x: List[float]):
        return f"Sum of list is {sum(x)}"

    agent = Agent(llm=setup_llm, tools=[])
    message = agent.invoke("Sum of this list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]?")
    response = message.content
    assert "55" in str(response), f"Expected '55' in response but got: {response}"


def test_agent_yfinance(setup_llm):
    import pandas as pd

    agent = Agent(
        description="You are a Financial Analyst",
        llm=setup_llm,
        skills=[
            "Deeply analyzing financial markets",
            "Searching information about stock price",
            "Visualization about stock price",
        ],
        tools=["longquanagent.tools.yfinance_tools"],
    )
    df = agent.invoke("What is the data of price of Tesla stock in 2024?")
    assert isinstance(df.artifact, pd.DataFrame)


def test_agent_visualization(setup_llm):
    from plotly.graph_objs._figure import Figure

    agent = Agent(
        description="You are a Financial Analyst",
        llm=setup_llm,
        skills=[
            "Deeply analyzing financial markets",
            "Searching information about stock price",
            "Visualization about stock price",
        ],
        tools=["longquanagent.tools.yfinance_tools"],
    )
    plot = agent.invoke("Let's visualize Tesla stock in 2024?")
    assert isinstance(plot.artifact, Figure)
