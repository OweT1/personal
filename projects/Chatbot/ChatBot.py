# Importing of Packages
import streamlit as st
from streamlit_chat import message
import time
from transformers import pipeline, AutoTokenizer
import torch

# General Settings
st.title("Hello There!")

chat_placeholder = st.empty()

owen_profile_pic = "Owen_Picture.jpg"

# Setting up of Transformer
torch.manual_seed(0)
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# User generator function
def response_generator(user_input):
  # Setting up of HuggingFace Model
  prompt = f"""Answer the question using the context below.
  Context: You are currently a Year 3 Data Science and Analytics student with a 2nd Major in Computer Science at National University of Singapore. You have a passion for Machine Learning, with wide interest in topics ranging from Classical Machine Learning concepts to Natural Language Processing and Neural Networks.
  Question: Can I have your Resume?
  Answer: Sure! You can take a look at my Resume [here](OwenTanKengLeng_Resume.pdf)
  Question: Can I have your Academic Transcript?
  Answer: Sure! You can take a look at my Academic Transcript [here](Owen_Transcript.pdf)
  Question: What are you passionate in?
  Answer: Great Question! I am passionate in a wide range of topics in Machine Learning, such as Neural Networks and Natural Language Processing concepts. That is why I decided to embark on this project to build a ChatBot! :)
  Question: {user_input}
  Answer:
  """

  sequences = pipe(
      prompt,
      max_new_tokens=10,
      do_sample=True,
      top_k=10,
      return_full_text = False,
  )
  for seq in sequences:
    response = seq["generated_text"]
    
  st.write_stream(stream_message(response))

# Response Streamer - Only for Bot
def stream_message(message):
  for word in message.split():
    yield word + " "
    time.sleep(0.05)

# Set up of Chat Interface & Initialisation of Chat History
welcome_message = "Hi There, I am Owen! Welcome to my chatbot :D"

with st.chat_message(name = "OwenBot", avatar = owen_profile_pic):
  st.write_stream(stream_message(welcome_message))

if "messages" not in st.session_state:
  # st.session_state.messages = [{"role": "OwenBot", "content": welcome_message}]
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(name = message["role"]):
    st.markdown(message["content"])

# User input and response
if user_input := st.chat_input("Ask me anything!"):
  with st.chat_message(name = "User"):
    st.markdown(user_input)
  st.session_state.messages.append({"role": "User", "content": user_input})
  
  with st.chat_message(name = "OwenBot", avatar = owen_profile_pic):
    response = response_generator(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

  

  
  
