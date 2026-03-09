# Name Generator

An MLP-based character-level language model trained to generate Indian names.



### 🔗 Live Demo
Check out the deployed app here: [https://name-generator-ris.streamlit.app/](https://name-generator-ris.streamlit.app/)

### 🛠️ Local Installation
```bash
git clone [https://github.com/yourusername/Name-Generator.git](https://github.com/yourusername/Name-Generator.git)
cd Name-Generator
pip install -r requirements.txt
```

### 🚀 Usage
```bash
streamlit run app.py
```

### 🧠 Model Logic
The model uses a 3-character context to predict the next character. It repeats until a . is predicted and filters for results >4 characters.
```
emb = C[context] # Embedding lookup
h = torch.tanh(emb.view(1, -1) @ W1 + b1) # Hidden layer
logits = h @ W2 + b2 # Output layer
```
