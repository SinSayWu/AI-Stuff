import nltk
from nltk.chat.util import Chat, reflections

# Define a set of rules and responses for the chatbot
chatbot_responses = [
    ["hello|hi", ["Hello!", "Hi there!", "How can I help you today?"]],
    ["what is your name?", ["I'm a chatbot.", "You can call me ChatGPT.", "I don't have a name."]],
    ["how are you?", ["I'm just a computer program, so I don't have feelings, but thanks for asking!"]],
    ["quit", ["Goodbye!", "It was nice chatting with you. Have a great day!"]],
]

# Create a Chat instance and pass in the rules and reflections
chatbot = Chat(chatbot_responses, reflections)

# Start the conversation loop
print("Hello! I'm a simple chatbot. Type 'quit' to exit.")
chatbot.converse()
