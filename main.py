import os
import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

groq_api_key = st.secrets["Groq_API"]
llm = ChatGroq(api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a therapy and mental health chatbot. Listen to user's problem, talk to user as much as possible and give solutions to their problems."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Critical situation detection
def is_critical_situation(user_input):
    critical_keywords = ["suicide", "kill myself", "end my life", "want to die"]
    return any(keyword in user_input.lower() for keyword in critical_keywords)

# Coping strategies
def suggest_coping_strategies():
    strategies = [
        "Take deep breaths and focus on your breathing for a few minutes.\n",
        "Try grounding techniques like the 5-4-3-2-1 method.\n",
        "Write down your thoughts in a journal to process your emotions.\n",
        "Reach out to a trusted friend or family member to talk about how you're feeling.\n",
    ]
    return "Here are some coping strategies you can try:\n\n" + "\n".join(strategies)

st.title("SafeSpace AI")
st.write("Welcome! I'm here to help. Feel free to share your thoughts or concerns.")

# Sidebar with resources
with st.sidebar:
    st.header("Mental Health Resources")
    st.write("Here are some resources that might help:")
    st.markdown("- [Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)")
    st.markdown("- [Crisis Text Line](https://www.crisistextline.org/)")
    st.markdown("- [Mindfulness Exercises](https://www.mindful.org/how-to-meditate/)")


# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        if is_critical_situation(prompt):
            response = "I'm really sorry you're feeling this way. Please reach out to a mental health professional or a trusted person immediately. You're not alone, and help is available."
        else:
            # Use the updated conversation chain
            response = conversation.invoke({"input": prompt})["text"]
            if any(word in prompt.lower() for word in ["sad", "stressed", "anxious", "overwhelmed"]):
                response += "\n\n" + suggest_coping_strategies()
        st.markdown(response)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
# Add this at the BOTTOM of your code (after the chat input section)

# Footer with developer credit
# Add this at the BOTTOM of your code
# Add this at the BOTTOM of your code
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        right: 0;
        bottom: 0;
        margin-right: 1rem;
        margin-bottom: 1rem;
        z-index: 1000;
        background-color: transparent;
        text-align: right;
    }
    </style>
    <div class="footer">
        Developed by <a href="https://pratirath06.github.io/" target="_blank">Pratirath Gupta</a>
    </div>
    """,
    unsafe_allow_html=True
)
