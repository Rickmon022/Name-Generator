import pickle as pk
import streamlit as st
import torch
import torch.nn.functional as F

# parameters loading
W1 = pk.load(open('./trained_parameters/W1.pkl', 'rb'))
W2 = pk.load(open('./trained_parameters/W2.pkl', 'rb'))
b1 = pk.load(open('./trained_parameters/b1.pkl', 'rb'))
b2 = pk.load(open('./trained_parameters/b2.pkl', 'rb'))
stoi = pk.load(open('./trained_parameters/stoi.pkl', 'rb'))
itos = pk.load(open('./trained_parameters/itos.pkl', 'rb'))
C = pk.load(open('./trained_parameters/C.pkl', 'rb'))

st.title('Indian Name Generator')

start_character = st.text_input("Write start character(s) of the name:", value="a").lower()
num_names = st.slider("Number of names to generate: ", 1,20,10)

if st.button('Generate'):
    block_size = 3
    names = []

    while len(names) < num_names:
        out = []
        context = [0] * block_size
        cnt_letters = 0
        for i, ch in enumerate(start_character):
            if ch in stoi:
                ix = stoi[ch]
                context = context[1:] + [ix]
                out.append(ix)

        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0 or len(out) >= 10:
                break

        name = ''.join(itos[i] for i in out).replace('.', '')
        if len(name) >= 4:
            names.append(name.capitalize())

    st.write(f"Names starting with '{start_character}' :")
    for name in names:
        st.write(f'-{name}')
