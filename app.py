import streamlit as st
from transformers import pipeline
import spacy_streamlit

@st.cache_resource
def load_nlp_pipelines():
    intent_pipe = pipeline(
        "text-classification", 
        model="philschmid/BERT-Banking77", 
        top_k=3 # Returns the top 3 intent probabilities instead of 1
    )
    
    ner_pipe = pipeline(
        "token-classification", 
        model="dslim/bert-base-NER", 
        aggregation_strategy="simple"
    )
    return intent_pipe, ner_pipe

def main():
    st.set_page_config(page_title="Intent & NER Engine", layout="centered")
    st.title("Intent Classification & Entity Extraction")

    intent_pipe, ner_pipe = load_nlp_pipelines()

    user_input = st.text_area(
        "Enter your query:", 
        "I tried to top up my app using my Chase card in Paris, but the bank rejected the 500 Euros transaction."
    )

    if st.button("Analyze Text"):
        if user_input.strip():
            with st.spinner("Processing text..."):
                # Execute inference
                intent_results = intent_pipe(user_input)
                ner_results = ner_pipe(user_input)

            # --- ENHANCEMENT 1: Top 3 Intent Probabilities ---
            st.subheader("Intent Classification (Top 3)")
            
            # intent_results is a list of lists when using top_k
            for result in intent_results[0]:
                intent = result['label']
                confidence = result['score']
                
                # Flag low confidence scores for human review
                if confidence < 0.70:
                    st.warning(f"**{intent}** (Confidence: {confidence:.2f}) - *Low Confidence*")
                else:
                    st.success(f"**{intent}** (Confidence: {confidence:.2f})")

            # --- ENHANCEMENT 2: Visual NER Highlighting ---
            st.subheader("Extracted Entities")
            if ner_results:
                # Reformat Hugging Face output into a dictionary spaCy can render
                docs = [{"text": user_input, "ents": [], "title": None}]
                
                for entity in ner_results:
                    docs[0]["ents"].append({
                        "start": entity["start"],
                        "end": entity["end"],
                        "label": entity["entity_group"]
                    })
                
                # Render the text with colorful entity highlights
                spacy_streamlit.visualize_ner(
                    docs,
                    labels=["LOC", "ORG", "PER", "MISC"],
                    show_table=False,
                    title="",
                    manual=True # Tells the visualizer we are providing custom entities
                )
            else:
                st.info("No named entities detected in the text.")

if __name__ == "__main__":
    main()