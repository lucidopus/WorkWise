import pandas as pd

from langchain import hub
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.chains import LLMChain
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.agents import Tool
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents import Tool, AgentExecutor

from langchain_tavily.tavily_search import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from utils.config import GROQ_API_KEY
from utils.prompts import TEMPLATE, VISUALIZATION_GENERATOR_PROMPT

from dotenv import load_dotenv
load_dotenv()

memory = ConversationBufferWindowMemory(k=6)

def get_company_info(company_name: str):
    """
    Args:
        company_name (str): The name of the company to get the rating and sentiment for.
    Returns:
        str: A string summarizing the company's overall rating and sentiment based on reviews.

    Use this to get information about a company. You should only use this if you need to look up information about a specific company
    """

    df = pd.read_csv("data/result.csv")

    # Check if the company exists in the DataFrame
    if company_name not in df["Company"].values:
        return f"Company '{company_name}' not found in the dataset."

    # Filter the data for the specific company
    company_data = df[df["Company"] == company_name]

    # Calculate the mean rating for WorkLifeBalance, CompensationBenefits, and Management
    mean_rating = (
        company_data[["WorkLifeBalance", "CompensationBenefits", "Management"]]
        .mean(axis=1)
        .iloc[0]
    )

    # Convert the mean rating to a normal float
    mean_rating = float(mean_rating)

    # Sentiment analysis based on mean rating
    sentiment = (
        "Positive"
        if mean_rating >= 4
        else "Neutral" if mean_rating == 3 else "Negative"
    )

    # Prepare the result
    result = {
        "Company": company_name,
        "OverallRating": mean_rating, 
    }
    return f"{company_name} has an overall rating of {mean_rating:.2f} and based on people's reviews, the sentiment is {sentiment}."


def pipeline(user_message: str):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    search_tool = TavilySearch(
        max_results=5,
        topic="general",
    )

    agent_tools = [
        Tool(
            name="tavily_search",
            func=search_tool.run,
            description="""Use this to answer general queries""",
        ),
        Tool(
            name="company_info",
            func=get_company_info,
            return_direct=True,
            description="""Use this tool when you need information about a specific company's employee sentiment analysis only""",
        ),
        Tool(
            name="salary_info",
            func = get_salary_data_info,
            description = """Use this only when someone asks about queries related to salaries. REMEMBER: You will be generating an input visualization prompt consisting of only these columns- Year,Company,Salary,Department,Overall_sentiment,Overall_experience"""
        ),
        Tool(
            name="get_job_related_data",
            func=get_job_related_data,
            description="""Use this only when someone asks about queries related to jobs and openings. REMEMBER: You will be generating an input visualization prompt consisting of only these columns- last_processed_time,job_title,company_name,job_location,first_seen_date,city,country,position,job_level,job_type"""
        )
    ]

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        callback_manager=callback_manager,
    )

    agent_tool_names = [tool.name for tool in agent_tools]
    agent_prompt = PromptTemplate(
        input_variables=["agent_scratchpad", "chat_history", "input"],
        partial_variables={
            "tools": render_text_description(agent_tools),
            "tool_names": ", ".join(agent_tool_names),
        },
        template=TEMPLATE,
    )
    agent = create_react_agent(llm, agent_tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        return_intermediate_steps=True,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
    )
    res = agent_executor.invoke(
        {
            "input": user_message,
            "chat_history": memory.buffer_as_str
        }
    )

    memory.save_context({'input': user_message}, {'output': res['output']})
    memory.load_memory_variables({})
    
    return res

def get_fig_from_code(code):
    local_variables = {}
    exec(code, {}, local_variables)
    return local_variables["fig"]

def get_job_related_data(visualization_prompt: str):
    """ The input must only include descriptions for creating plotly visualization that best answers the user's question consisting of the following columns: ['last_processed_time', 'job_title', 'company_name', 'job_location',
       'first_seen_date', 'city', 'country', 'position', 'job_level',
       'job_type']
    """

    data_path = "jobs.csv"  # Path to the CSV file containing job data
    df = pd.read_csv(data_path)
    print(df)
    csv_string = df.head().to_string(index=False)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VISUALIZATION_GENERATOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="text",
        verbose=True,
    )
    res = chain.run(
        {"messages": [HumanMessage(content=visualization_prompt)], "data": csv_string, 'filename': data_path}
    )

    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", res, re.DOTALL)

    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r"(?m)^\s*fig\.show\(\)\s*$", "", code_block)
        local_variables = {}
        exec(cleaned_code, {}, local_variables)
        fig = local_variables["fig"]
        return fig
    else:
        return "NO PLOTS GENERATED"
    
def get_employee_satisfaction_related_data(visualization_prompt: str):
    data_path = "employee.csv" 
    df = pd.read_csv(data_path)
    csv_string = df.head().to_string(index=False)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VISUALIZATION_GENERATOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="text",
        verbose=True,
    )
    res = chain.run(
        {"messages": [HumanMessage(content=visualization_prompt)], "data": csv_string, 'filename': data_path}
    )

    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", res, re.DOTALL)

    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r"(?m)^\s*fig\.show\(\)\s*$", "", code_block)
        local_variables = {}
        exec(cleaned_code, {}, local_variables)
        fig = local_variables["fig"]
        return fig
    else:
        return "NO PLOTS GENERATED"



def get_salary_data_info(visualization_prompt: str):
    """ The input must only include descriptions for creating plotly visualization that best answers the user's question consisting of the following columns: [Year,Company,Salary,Department,Overall_sentiment,Overall_experience]
    """

    data_path = "compensation.csv"  # Path to the CSV file containing job data
    df = pd.read_csv(data_path)
    print(df)
    csv_string = df.head().to_string(index=False)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VISUALIZATION_GENERATOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="text",
        verbose=True,
    )
    res = chain.run(
        {"messages": [HumanMessage(content=visualization_prompt)], "data": csv_string, 'filename': data_path}
    )

    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", res, re.DOTALL)

    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r"(?m)^\s*fig\.show\(\)\s*$", "", code_block)
        local_variables = {}
        exec(cleaned_code, {}, local_variables)
        fig = local_variables["fig"]
        return fig
    else:
        return "NO PLOTS GENERATED"
    
def get_employee_satisfaction_related_data(visualization_prompt: str):
    data_path = "employee.csv" 
    df = pd.read_csv(data_path)
    csv_string = df.head().to_string(index=False)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VISUALIZATION_GENERATOR_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="text",
        verbose=True,
    )
    res = chain.run(
        {"messages": [HumanMessage(content=visualization_prompt)], "data": csv_string, 'filename': data_path}
    )

    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", res, re.DOTALL)

    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r"(?m)^\s*fig\.show\(\)\s*$", "", code_block)
        local_variables = {}
        exec(cleaned_code, {}, local_variables)
        fig = local_variables["fig"]
        return fig
    else:
        return "NO PLOTS GENERATED"
