import streamlit as st
from pathlib import Path
import re
import time
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fact_check import fact_check
from news_api import search_news

st.set_page_config(page_title="TruthGuard AI",page_icon="🛡️",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>

.stApp{
    background:#0B1220;
}

.block-container{
    max-width:900px;
    padding-top:2rem;
}

h1,h2,h3{
    color:#F8FAFC;
}

p,label{
    color:#CBD5E1;
}

.stTextArea textarea{
    background:#111827;
    color:white;
    border:1px solid #334155;
    border-radius:12px;
}

.stButton>button{
    width:100%;
    height:22px;
    border-radius:10px;
    background:#2563EB;
    color:white;
    border:none;
    font-weight:200;
    font-size:16px;
}

.stButton>button:hover{
    background:#1D4ED8;
}

.result-card{
    background:#111827;
    border:1px solid #334155;
    border-radius:8px;
    padding:8px;
    margin-top:10px;
}

.metric-card{
    background:#1E293B;
    border:1px solid #334155;
    border-radius:6px;
    padding:10px;
    text-align:center;
}

.badge-real{
    background:#16A34A;
    color:white;
    padding:10px 18px;
    border-radius:10px;
    font-weight:500;
    display:inline-block;
}

.badge-fake{
    background:#DC2626;
    color:white;
    padding:10px 18px;
    border-radius:10px;
    font-weight:500;
    display:inline-block;
}

.badge-uncertain{
    background:#D97706;
    color:white;
    padding:10px 18px;
    border-radius:10px;
    font-weight:500;
    display:inline-block;
}

.badge-standalone{
    background:#1E293B;
    border-left:4px solid #64748B;
    padding:15px;
    border-radius:10px;
    color:#CBD5E1;
}

div[data-testid="stMetric"]{
    background:#111827;
    border:1px solid #334155;
    padding:12px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def default_paths():
    root = Path(__file__).resolve().parents[1]
    return {
        "bert_model": root / "outputs" / "bert_model"
    }


@st.cache_resource
def load_bert(model_path):
    if not model_path.exists():
        return None, None

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    return tokenizer, model


def predict_bert(text, tokenizer, model):

    inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    
    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    return real_prob, fake_prob


def main():

    
    tokenizer, bert_model = load_bert(default_paths()["bert_model"])

    if tokenizer is None:
        st.error("DistilBERT model not found.")
        return

    
    with st.sidebar:
        st.markdown("## TruthGuard AI")
        st.markdown("### Hybrid Fake News Detection")
        st.write(
            "DistilBERT predicts whether a news article appears real or fake, "
            "while Google Fact Check verifies if the claim has already been reviewed."
        )
        
        st.divider()

        
        with st.expander("⚙️ Advanced Settings", expanded=False):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.20,
                max_value=0.80,
                value=0.50,
                step=0.05
            )

    
    st.title("")
    st.caption("Paste any headline or article below.")

    
    st.markdown("Try a sample headline:")
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("Stock", use_container_width=True):
            st.session_state.news_input = "Market manipulation suspected after sudden stock crash."
    with col_btn2:
        if st.button("Alien", use_container_width=True):
            st.session_state.news_input = "Secret files leak showing proof of alien spacecraft in Area 51."
    with col_btn3:
        if st.button("Health", use_container_width=True):
            st.session_state.news_input = "Drinking lemon juice daily completely cures modern chronic diseases."

    
    if 'news_input' not in st.session_state:
        st.session_state.news_input = ""

    news_text = st.text_area(
        "News", 
        value=st.session_state.news_input, 
        placeholder="Type or paste the news headline here...", 
        height=100,
        label_visibility="collapsed"
    )

    
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    
    if analyze_clicked:
        if not news_text.strip():
            st.warning("Enter some news.")
            return

        
        with st.spinner("Analyzing..."):
            start = time.perf_counter()

            real_prob, fake_prob = predict_bert(
                clean_text(news_text),
                tokenizer,
                bert_model
            )

            confidence = max(real_prob, fake_prob) * 100
            elapsed = time.perf_counter() - start

            if real_prob >= confidence_threshold:
                label = "REAL"
            elif fake_prob >= confidence_threshold:
                label = "FAKE"
            else:
                label = "UNCERTAIN"

            facts = fact_check(news_text)

            query = " ".join(news_text.split()[:8])
            news_articles = search_news(query)


            label = "UNCERTAIN"


            if facts:

                rating = facts[0]["rating"].lower()

                if any(x in rating for x in [
                    "false", "fake", "incorrect", "pants on fire"
                ]):
                    label = "FAKE"

                elif any(x in rating for x in [
                    "true", "correct", "accurate"
                ]):
                    label = "REAL"


            elif len(news_articles) >= 3 and real_prob >= 0.90:
                label = "REAL"


            elif fake_prob >= 0.95:
                label = "FAKE"

            elif real_prob >= 0.95:
                label = "REAL"


            else:
                label = "UNCERTAIN"

        st.divider()

            
        if label == "REAL":
            st.markdown('<div class="badge-real">✅ REAL NEWS</div>', unsafe_allow_html=True)

        elif label == "FAKE":
            st.markdown('<div class="badge-fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="badge-uncertain">⚠ UNCERTAIN</div>', unsafe_allow_html=True)

        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Confidence")
            st.write(f"**Real:** {real_prob * 100:.1f}%")
            st.progress(real_prob)
            st.write(f"**Fake:** {fake_prob * 100:.1f}%")
            st.progress(fake_prob)
            
            st.metric("Overall Confidence", f"{confidence:.1f}%")
            st.caption(f"Inference Time: {elapsed:.2f} sec")
            
        with col2:
            st.subheader("💡 Explanation")
            reasons = []
            if real_prob > 0.75:
                reasons.append("Language resembles credible journalism.")
            elif fake_prob > 0.75:
                reasons.append("Language resembles misinformation patterns.")
            else:
                reasons.append("Prediction confidence is moderate.")

            clickbait = ["breaking", "shocking", "urgent", "viral", "exclusive"]
            found = [w for w in clickbait if w in news_text.lower()]
            if found:
                reasons.append("Clickbait words detected: " + ", ".join(found))
            if len(news_text.split()) < 6:
                reasons.append("Very short text reduces accuracy.")
            if "!" in news_text:
                reasons.append("Emotional punctuation detected.")

            for r in reasons:
                st.write(f"• {r}")
            
        with col3:
            st.subheader("🔍 Google Fact Check")
            if facts:
                for fact in facts:
                    st.markdown(f"""
                ### ✅ Fact Check

                **Claim:** {fact['text']}

                **Rating:** {fact['rating']}

                **Publisher:** {fact['publisher']}
                """)
            else:
                st.warning("⚠️ No verified claim found. Result based only on the AI model.")

        st.divider()

        st.subheader("📰 Latest News")

        if news_articles:
            for article in news_articles[:3]:
                st.markdown(f"""
        ### 📰 {article['title']}

        **Source:** {article['source']['name']}

        **Published:** {article['publishedAt'][:10]}
                """)
        else:
            st.warning("No related news found.")  
    
        st.divider()
    
        st.subheader("About TruthGuard AI")
        st.write("""
        TruthGuard AI is a hybrid fake news detection system.
        
        • **DistilBERT** analyzes the language patterns of the news article.
        
        • **Google Fact Check** searches trusted fact-checking organizations.

        • **NewsAPI** searches trusted news published recently.
        
        if neither service finds anything, DistilBERT makes the prediction.
        """)

        st.caption("Built with Streamlit • DistilBERT • Google Fact Check API * NewsAPI")

if __name__ == "__main__":
    main()