import streamlit as st 
import re
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

# Fonction pour extraire le score de similarité de la réponse générée par GPT-3
def extract_score(response):
    match = re.search(r"\d+\.\d+", response)
    if match:
        return match.group()
    else:
        return "Pas de similarité trouvé."

# Fonction pour récupérer la clé API OpenAI GPT-3 saisie par l'utilisateur
def get_api_key():
    api_key = st.text_input("Entrez votre clé OpenAI:")
    return api_key

# Fonction pour récupérer la similarité entre deux textes en utilisant l'API OpenAI GPT-3

def get_similarity(text1, text2, model_engine, api_key):
    openai.api_key = api_key
    # Choose an embedding model
    model_id = "text-similarity-davinci-001"
    # Compute embeddings of the two texts
    embeddings = openai.Embedding.create(inputs=[text1, text2], model=model_id)["data"]
    # Compute the cosine similarity between the embeddings
    similarity_score = cosine_similarity(embeddings[0]["embedding"], embeddings[1]["embedding"])
    return similarity_score
    
# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Similarité entre textes")
    api_key = get_api_key()
    if api_key:
        model_engine = "de-code-davinci-002"
        text1 = st.text_area("Texte 1")
        text2 = st.text_area("Texte 2")
        if st.button("Compare"):
            similarity_score = get_similarity(text1, text2, model_engine, api_key)
            st.write(f"Le score de similarité entre les deux textes est {similarity_score}.")

if __name__ == "__main__":
    main()
