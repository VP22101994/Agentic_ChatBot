from langgraph.graph import StateGraph
from src.langgraphagenticai.state.state import State
from langgraph.graph import START,END
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.tools.search_tool import get_tools,create_tool_node
from langgraph.prebuilt import tools_condition,ToolNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
import networkx as nx
import matplotlib.pyplot as plt
from src.langgraphagenticai.nodes.ai_news_node import AINewsNode

def save_graph_with_networkx(graph_builder, filename="graph_image.png"):
    G = nx.DiGraph()

    # Add nodes
    for node in graph_builder.nodes:
        G.add_node(node)

    # Add edges
    for source, target in graph_builder.edges:
        G.add_edge(source, target)

    # Draw and save the graph
    pos = nx.spring_layout(G)  # or use nx.shell_layout(G), etc.
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    plt.title("LangGraph Flow")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Graph saved as {filename}")

def save_langgraph_graph(builder, filename="graph.png"):
    
    G = nx.DiGraph()
    # internal_graph = builder  # Only available before compile

    for node in builder.nodes:
        G.add_node(node)

    for source, target in builder.edges:
        G.add_edge(source, target)

    # Draw
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, arrows=True)
    plt.title("LangGraph Workflow")
    plt.axis("off")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Graph saved as {filename}")




class GraphBuilder:
    def __init__(self,model):
        self.llm=model
        self.graph_builder=StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """

        self.basic_chatbot_node=BasicChatbotNode(self.llm)

        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)

        save_graph_with_networkx(self.graph_builder, "basic_chatbot_graph")


    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node
        tools=get_tools()
        tool_node=create_tool_node(tools)

        ## Define the LLM
        llm=self.llm

        ## Define the chatbot node

        obj_chatbot_with_node=ChatbotWithToolNode(llm)
        chatbot_node=obj_chatbot_with_node.create_chatbot(tools)
        ## Add nodes
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools",tool_node)
        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")

        save_langgraph_graph(self.graph_builder, "chatbot_graph.png")

    def ai_news_builder_graph(self):

        ai_news_node=AINewsNode(self.llm)

        ## added the nodes

        self.graph_builder.add_node("fetch_news",ai_news_node.fetch_news)
        self.graph_builder.add_node("summarize_news",ai_news_node.summarize_news)
        self.graph_builder.add_node("save_result",ai_news_node.save_result)

        #added the edges

        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news","summarize_news")
        self.graph_builder.add_edge("summarize_news","save_result")
        self.graph_builder.add_edge("save_result", END)

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        if usecase == "Chatbot With Web":
            self.chatbot_with_tools_build_graph()
        if usecase == "AI News":
            self.ai_news_builder_graph()

        return self.graph_builder.compile()
