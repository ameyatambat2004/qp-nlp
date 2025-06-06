import streamlit as st
import pickle
import helper as hp
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from sentence_transformers import SentenceTransformer, util

model_bert = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(q1, q2):
    embeddings = model_bert.encode([q1, q2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())

model_base = pickle.load(open("D:\Ameya\Coding\Machine Learning\Projects\Question Pair\model_base.pkl",'rb'))
cv_base = pickle.load(open('D:\Ameya\Coding\Machine Learning\Projects\Question Pair\cv_base.pkl','rb'))

model_adv = pickle.load(open("D:\Ameya\Coding\Machine Learning\Projects\Question Pair\model_adv.pkl",'rb'))
cv_adv = pickle.load(open("D:\Ameya\Coding\Machine Learning\Projects\Question Pair\cv_adv.pkl",'rb'))

model_lstm = load_model("D:\Ameya\Coding\Machine Learning\Projects\Question Pair\qpair-lstm-model.h5")
myvectorizer = pickle.load(open("D:\Ameya\Coding\Machine Learning\Projects\Question Pair\myvectorizer.pkl",'rb'))


st.header('Question Pairs Similarity')
    

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')


if st.button('Find'):
    st.write('### Similarity (BERT-based Model) :', round(semantic_similarity(q1, q2),2))    
    clear_session()
    st.write('### Similartiy (LSTM Model) :', round(model_lstm.predict(hp.query_point_creator_lstm(q1, q2, myvectorizer))[0][0],2))
    st.write('### Similarity (Adv Model) :', model_adv.predict_proba(hp.query_point_creator(q1, q2, cv_adv))[0][1])
    st.write('### Similarity (Base Model) :', hp.predict_similarity(q1, q2, cv_base, model_base))
    with st.expander("More info"):
        st.write("BERT-based Model : (only for comparison) This is a pretrained model which uses the BERT-based all-MiniLM-L6-v2 sentence transformer to encode both questions into dense vectors and calculates cosine similarity between them.")
        st.write("LSTM Model : This functional model is a deep neural network with Embeddings, Bidirectional LSTMs, Dense and Dropout Layers. It was trained on ~400000 rows and val_accuracy of 81.01%.")
        st.write("Adv Model : This is a basic Bag of Words + Random Forest based model with custom Features and Preprocessing. It was trained 30000 rows and 77.72% accuracy.")
        st.write("Base Model : This is the baseline model with no preprocessing or feature engineering. Based on Bag of Words + Random Forest, trained on 40000 rows with 75.5% accuracy.")
    
    
# Footer with profile links
st.markdown('---')
col1, col2, col3 = st.columns([1, 1, 6])
with col1:
    st.markdown('<a href="https://linkedin.com/in/your-profile" target="_blank"><img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" width="40" style="filter: invert(1)" class="dark-logo"></a><style>.dark-logo{filter:invert(0)!important}@media (prefers-color-scheme:dark){.dark-logo{filter:invert(1)!important}}</style>', unsafe_allow_html=True)
with col2:
    st.markdown('<a href="https://github.com/your-username" target="_blank"><img src="https://img.icons8.com/ios-filled/50/000000/github.png" width="40" style="filter: invert(1)" class="dark-logo"></a>', unsafe_allow_html=True)