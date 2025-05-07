import pytest
from langchain_together import ChatTogether
from longquanagent.agent.agent import Agent
from langchain_core.messages import ToolMessage
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

    agent = Agent(
        agent_name="Normal",
        llm=setup_llm,
        tools=[])
    message = agent.invoke("Sum of this list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]?")
    response = message.content
    assert "55" in str(response), f"Expected '55' in response but got: {response}"


def test_agent_sum(setup_llm):
    from typing import List

    agent = Agent(
        agent_name="Sum Agent",
        description="You can sum a list",
        llm=setup_llm,
        skills=[
            "You can calculate a list and array of numbers",
        ],
        agent_template="agent_template/agent_template.json",
    )

    @agent.function_tool
    def sum_of_series(x: List[float]):
        """Sum of series
        Args:
            - x: List of numbers
        Returns:
            - Sum of series
        """
        return f"Sum of list is {sum(x)}"

    message = agent.invoke("Sum of this list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]?")
    assert isinstance(message, ToolMessage) and ("55" in message.artifact)


def test_absolute_path(setup_llm):
    agent = Agent(
        agent_name="Hello Agent",
        description="You can say hello to people",
        llm=setup_llm,
        skills=[
            "You say hello",
        ],
        tools=["/Users/phamdinhkhanh/Documents/Courses/Manus/longquanagent/hello.py"],
        agent_template="agent_template/agent_template.json",
    )
    message = agent.invoke("Can you say hello to me?")
    assert isinstance(message, ToolMessage)


def test_agent_yfinance(setup_llm):
    import pandas as pd

    agent = Agent(
        agent_name="Financial Analyst",
        description="You are a Financial Analyst",
        llm=setup_llm,
        skills=[
            "Deeply analyzing financial markets",
            "Searching information about stock price",
            "Visualization about stock price",
        ],
        tools=["longquanagent.tools.yfinance_tools"],
        agent_template="agent_template/agent_template.json",
    )
    df = agent.invoke("What is the data of price of Tesla stock in 2024?")
    print("df: ", df)
    # assert isinstance(df.artifact, pd.DataFrame)


def test_agent_visualization(setup_llm):
    from plotly.graph_objs._figure import Figure

    agent = Agent(
        agent_name="Visualize Analyst",
        description="You are a Visualize Analyst",
        llm=setup_llm,
        skills=[
            "Deeply analyzing financial markets",
            "Searching information about stock price",
            "Visualization about stock price",
        ],
        tools=["longquanagent.tools.yfinance_tools"],
        agent_template="agent_template/agent_template.json",
    )
    plot = agent.invoke("Let's visualize Tesla stock in 2024?")
    print("plot: ", plot)
    assert isinstance(plot.artifact, Figure)
