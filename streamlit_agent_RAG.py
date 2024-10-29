import streamlit as st
import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

# Set up the environment and agent
os.environ['OPENAI_API_KEY'] = ""

# Settings
Settings.llm = Ollama(model="mixtral", request_timeout=120.0, temperature=0.4)

# Function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# RAG pipeline
documents = SimpleDirectoryReader("/home/sanka/NLP/IOT_knowledge_base/data2").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# RAG pipeline as a tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine, 
    name="ESTCP_EW_Final_Report_Document",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget."
)

# Initialize the agent
agent = ReActAgent.from_tools([multiply_tool, add_tool, budget_tool])

# Streamlit app
def main():
    st.title("USE case design assitant for Group NIRE and GLEAMM IoNM test bed")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("You:")

    if st.button("Send"):
        if user_input:
            st.session_state.conversation.append(("You", user_input))
            try:
                response = agent.chat(user_input)
                if isinstance(response, dict) :
                    st.session_state.conversation.append(("Design Assist:", response))
                else:
                    st.session_state.conversation.append(("Design Assist:", str(response)))
            except Exception as e:
                st.session_state.conversation.append(("Design Assist:", f"An error occurred: {e}"))

    st.write("## Conversation")
    for speaker, text in st.session_state.conversation:
        st.write(f"**{speaker}:** {text}")

if __name__ == "__main__":
    main()