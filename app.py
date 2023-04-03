import streamlit as st 
import re
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

# Fonction pour récupérer la clé API OpenAI GPT-3 saisie par l'utilisateur
def get_api_key():
    api_key = st.text_input("Entrez votre clé OpenAI:")
    return api_key

# Fonction pour récupérer les embeddings de deux textes en utilisant l'API OpenAI GPT-3
def get_embeddings(text1, text2, model_engine, api_key):
    openai.api_type = "azure"
    openai.api_key = api_key
    openai.api_base = "https://tsi-openai.openai.azure.com"
    openai.api_version = "2022-12-01"

    embeddings = openai.Embedding.create(
        inputs=[text1, text2],
        model=model_engine,
        data_output_format="array",
        engine="davinci"
    )["data"]

    return embeddings

# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Similarité entre textes")
    api_key = get_api_key()
    if api_key:
        model_engine = "text-embedding-ada-002"
        text1 = st.text_area("Texte 1")
        text2 = st.text_area("Texte 2")
        if st.button("Compare"):
            embeddings = get_embeddings(text1, text2, model_engine, api_key)
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            st.write(f"Le score de similarité de cosinus entre les deux textes est {similarity:.4f}.")

if __name__ == "__main__":
    main()
