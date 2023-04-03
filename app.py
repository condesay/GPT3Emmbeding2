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
    openai.api_type = "azure"
    openai.api_key = api_key
    openai.api_base = "https://tsi-openai.openai.azure.com"
    openai.api_version = "2022-12-01"
    # create a completion
    response = openai.Completion.create(
        deployment_id= model_engine, 
        prompt=f"Compare the similarity between these two texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSimilarity:",
        max_tokens=64,
        temperature=0.5
    )
    similarity = response.choices[0].text.strip()
    similarity_score = extract_score(similarity)
    return similarity_score

# Fonction pour récupérer les embeddings de deux textes en utilisant l'API OpenAI GPT-3
def get_embeddings(text1, text2, model_id, api_key):
    openai.api_key = api_key
    embeddings = openai.Embedding.create(inputs=[text1, text2], model=model_id, data_output_format="array")["data"]
    return embeddings

# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Similarité entre textes")
    api_key = get_api_key()
    if api_key:
        model_engine = "de-code-davinci-002"
        model_id = "text-similarity-davinci-002"
        text1 = st.text_area("Texte 1")
        text2 = st.text_area("Texte 2")
        if st.button("Compare"):
            similarity_score = get_similarity(text1, text2, model_engine, api_key)
            embeddings = get_embeddings(text1, text2, model_id, api_key)
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            st.write(f"Le score de similarité entre les deux textes est {similarity_score}.")
            st.write(f"Le score de similarité de cosinus entre les deux textes est {similarity:.4f}.")

if __name__ == "__main__":
    main()
