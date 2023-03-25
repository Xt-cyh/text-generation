from pychatgpt import Chat


# Initializing the chat class will automatically log you in, check access_tokens
email = 'yihanchen0517@gmail.com'
passwd = '_ktG.i8_HfTqr.Z'
chat = Chat(email=email, password=passwd) 
answer = chat.ask("Hello!")