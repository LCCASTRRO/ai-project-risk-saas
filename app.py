import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="AI Project Risk", layout="wide")

st.title("🚀 AI Project Risk Dashboard")

@st.cache_data
def gerar_dados():
    rng = np.random.default_rng(42)
    n = 1000

    df = pd.DataFrame({
        "Duracao_Planeada": rng.integers(30,240,n),
        "Orcamento_Planeado": rng.integers(8000,250000,n),
        "Numero_Recursos": rng.integers(2,15,n),
        "Complexidade": rng.integers(1,6,n),
        "Risco_Inicial": rng.integers(1,6,n),
        "Mudancas_Escopo": rng.poisson(1.2,n),
        "Percentagem_Conclusao_50pct": rng.normal(52,12,n).clip(5,95)
    })

    df["Atraso"] = rng.integers(0,2,n)
    return df

df = gerar_dados()

X = df.drop("Atraso", axis=1)
y = df["Atraso"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = RandomForestClassifier()
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:,1]

st.subheader("📊 Performance do Modelo")
st.write("ROC-AUC:", round(roc_auc_score(y_test, proba),3))

st.divider()
st.subheader("🔎 Simulador de Projeto")

dur = st.number_input("Duração Planeada", 30,400,120)
orc = st.number_input("Orçamento Planeado",1000,500000,50000)
rec = st.number_input("Nº Recursos",1,50,6)
comp = st.slider("Complexidade",1,5,3)
risco = st.slider("Risco Inicial",1,5,3)
mud = st.number_input("Mudanças de Escopo",0,20,1)
pct = st.slider("Percentagem Concluída a Meio",0,100,50)

novo = pd.DataFrame([{
    "Duracao_Planeada":dur,
    "Orcamento_Planeado":orc,
    "Numero_Recursos":rec,
    "Complexidade":comp,
    "Risco_Inicial":risco,
    "Mudancas_Escopo":mud,
    "Percentagem_Conclusao_50pct":pct
}])

prob = model.predict_proba(novo)[0][1]

st.metric("Probabilidade de Atraso", f"{prob:.1%}")

if prob > 0.5:
    st.error("🔴 Projeto em Risco")
else:
    st.success("🟢 Projeto Sob Controlo")
