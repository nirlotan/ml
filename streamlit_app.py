import streamlit as st
from empath import Empath




st.title("Sentiment Analysis Demo")
input = st.text_input("Your sentence:","Write a sentence here")

lexicon = Empath()
d = lexicon.analyze(input)

res = ""
for key, value in d.items():
    if value > 0:
        res += key + ", "
        
st.text("Sentiments in the sentence:")
st.write(res[:-2])

