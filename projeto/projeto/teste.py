import pandas as p
import streamlit as st

@st.cache_data
def load_data():
    try:
        return p.read_csv("albums.csv", sep="t")
    except FileNotFoundError:
        st.error("O arquivo 'albums.csv' não foi encontrado. Verifique o caminho e tente novamente.")
        return p.DataFrame()

data = load_data()

if not data.empty:
    st.write("Dataset Musicoset Metadata", data)

    st.write("Gráfico de Exemplo")
    chart_data = data.groupby("popularity")[""].mean().reset_index()
    st.bar_chart(chart_data.set_index("genre"))
else:
    st.write("Não foi possível carregar os dados.")

