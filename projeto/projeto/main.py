import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Análise de Dados do Spotify",
    layout="wide"
)

@st.cache_data
def load_data():
    """Carrega os dados do arquivo CSV"""
    csv_path = Path(__file__).parent.parent / "SpotifyFeatures.csv"
    try:
        data = pd.read_csv(csv_path)
        return data
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {csv_path}")
        return None

@st.cache_data
def calculate_top_artists(genre_filter):
    """Calcula top artistas com cache para melhor performance"""
    data = load_data()
    if data is None:
        return pd.DataFrame()
    
    # Aplicar filtro se necessário
    if genre_filter != 'Todos':
        # Filtrar considerando as variações do gênero
        selected_normalized = re.sub(r'[^a-zA-Z0-9\s]', '', genre_filter.lower())
        mask = data['genre'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()) == selected_normalized)
        filtered_data = data[mask]
    else:
        # Quando "Todos", pegar apenas os 500 artistas mais populares para melhor performance
        artist_max_popularity = data.groupby('artist_name')['popularity'].max()
        top_artists = artist_max_popularity.nlargest(500).index
        filtered_data = data[data['artist_name'].isin(top_artists)]
    
    if len(filtered_data) == 0:
        return pd.DataFrame()
    
    # Otimização: usar groupby e nlargest mais eficientemente
    # Primeiro, ordenar por artista e popularidade
    sorted_data = filtered_data.sort_values(['artist_name', 'popularity'], ascending=[True, False])
    
    # Pegar as top 3 músicas de cada artista de forma mais eficiente
    top_3_per_artist = sorted_data.groupby('artist_name').head(3)
    
    # Calcular estatísticas por artista
    artist_stats = top_3_per_artist.groupby('artist_name').agg({
        'popularity': ['mean', 'max'],
        'track_name': 'first'  # primeira música (mais popular)
    }).round(1)
    
    # Achatar colunas multi-nível
    artist_stats.columns = ['avg_top3_popularity', 'best_song_popularity', 'best_song']
    artist_stats = artist_stats.reset_index()
    
    # Adicionar contagem total de músicas por artista
    total_songs = filtered_data['artist_name'].value_counts().to_dict()
    artist_stats['total_songs'] = artist_stats['artist_name'].map(total_songs)
    
    # Pegar apenas os top 10
    top_artists = artist_stats.nlargest(10, 'avg_top3_popularity')
    
    return top_artists

def main():
    st.title("🎵 Análise de Dados do Spotify Features")
    st.markdown("Uma análise exploratória das características musicais no Spotify")
    
    # Carregando os dados
    data = load_data()
    if data is None:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo SpotifyFeatures.csv existe.")
        return
    
    # Filtros na barra lateral
    st.sidebar.header("Filtros")
    
    # Filtro por gênero - removendo duplicatas e limpeza
    unique_genres = data['genre'].unique()
    # Limpar e normalizar gêneros para remover duplicatas
    cleaned_genres = []
    seen_genres = set()
    
    for genre in unique_genres:
        # Normalizar: remover todos os tipos de aspas, espaços e caracteres especiais
        normalized = re.sub(r'[""\'`]', "'", genre.strip())
        normalized_key = re.sub(r'[^a-zA-Z0-9\s]', '', normalized.lower())
        
        if normalized_key not in seen_genres:
            seen_genres.add(normalized_key)
            cleaned_genres.append(normalized)
    
    genres = ['Todos'] + sorted(cleaned_genres)
    selected_genre = st.sidebar.selectbox("Selecione um gênero:", genres)
    
    # Filtrar dados se necessário
    filtered_data = data.copy()
    if selected_genre != 'Todos':
        # Criar um mapeamento para filtrar considerando as variações do gênero
        selected_normalized = re.sub(r'[^a-zA-Z0-9\s]', '', selected_genre.lower())
        mask = data['genre'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()) == selected_normalized)
        filtered_data = data[mask]
    
    # Informações gerais
    st.header("📊 Visão Geral dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Músicas", len(filtered_data))
    with col2:
        st.metric("Total de Artistas", filtered_data['artist_name'].nunique())
    with col3:
        st.metric("Total de Gêneros", data['genre'].nunique())
    with col4:
        if selected_genre != 'Todos':
            avg_popularity = filtered_data['popularity'].mean()
            st.metric("Popularidade Média", f"{avg_popularity:.1f}")
        else:
            avg_popularity_all = data['popularity'].mean()
            st.metric("Popularidade Média", f"{avg_popularity_all:.1f}")
    
    # 1. Distribuição por gênero
    st.subheader("🎭 Distribuição por Gênero")
    
    genre_counts = data['genre'].value_counts()
    
    fig_genre = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Distribuição de Músicas por Gênero",
        labels={'x': 'Gênero', 'y': 'Quantidade de Músicas'},
        color=genre_counts.values,
        color_continuous_scale='viridis'
    )
    fig_genre.update_layout(
        xaxis_title="Gênero Musical",
        yaxis_title="Quantidade de Músicas",
        showlegend=False,
        height=400
    )
    fig_genre.update_xaxes(tickfont=dict(size=9))
    st.plotly_chart(fig_genre, use_container_width=True)
    
    # 2. Distribuição de popularidade
    st.subheader("⭐ Análise de Popularidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pop = px.histogram(
            filtered_data,
            x='popularity',
            nbins=20,
            title="Distribuição da Popularidade",
            labels={'popularity': 'Popularidade', 'count': 'Frequência'}
        )
        st.plotly_chart(fig_pop, use_container_width=True)
    
    with col2:
        # Top 10 gêneros por popularidade média
        if selected_genre == 'Todos':
            genre_popularity = data.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10)
            fig_genre_pop = px.bar(
                x=genre_popularity.values,
                y=genre_popularity.index,
                orientation='h',
                title="Top 10 Gêneros por Popularidade Média",
                labels={'x': 'Popularidade Média', 'y': 'Gênero'}
            )
            st.plotly_chart(fig_genre_pop, use_container_width=True)
        else:
            # Mostrar características do gênero selecionado
            st.write(f"**Características do gênero: {selected_genre}**")
            features = ['danceability', 'energy', 'valence', 'acousticness']
            genre_means = filtered_data[features].mean()
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=genre_means.values,
                theta=genre_means.index,
                fill='toself',
                name=selected_genre
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Perfil Musical do Gênero"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # 3. Análise de duração
    st.subheader("⏱️ Análise de Duração")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição da duração
        duration_minutes = filtered_data['duration_ms'] / 60000
        fig_duration = px.histogram(
            x=duration_minutes,
            nbins=30,
            title="Distribuição da Duração das Músicas (minutos)"
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Duração média por gênero
        duration_by_genre = data.groupby('genre')['duration_ms'].mean().sort_values(ascending=False).head(10)
        duration_by_genre_min = duration_by_genre / 60000
        
        fig_duration_genre = px.bar(
            x=duration_by_genre_min.values,
            y=duration_by_genre_min.index,
            orientation='h',
            title="Duração Média por Gênero (minutos)"
        )
        st.plotly_chart(fig_duration_genre, use_container_width=True)
    
    # 4. Análise de artistas
    st.subheader("🎤 Análise de Artistas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top artistas mais populares (música individual - apenas uma por artista)
        if len(filtered_data) > 0:
            try:
                # Se o filtro for "Todos", usar o dataset completo
                if selected_genre == 'Todos':
                    # Pegar apenas a música mais popular de cada artista
                    top_tracks = data.loc[data.groupby('artist_name')['popularity'].idxmax()].nlargest(10, 'popularity')[['artist_name', 'track_name', 'popularity']]
                else:
                    # Para gêneros específicos, usar filtered_data
                    top_tracks = filtered_data.loc[filtered_data.groupby('artist_name')['popularity'].idxmax()].nlargest(10, 'popularity')[['artist_name', 'track_name', 'popularity']]
                
                fig_top_tracks = px.bar(
                    top_tracks,
                    x='popularity',
                    y='artist_name',
                    hover_data=['track_name'],
                    orientation='h',
                    title="Top 10 Artistas (Música Mais Popular)",
                    labels={'popularity': 'Popularidade', 'artist_name': 'Artista'},
                    color='popularity',
                    color_continuous_scale='Reds'
                )
                fig_top_tracks.update_layout(height=400)
                st.plotly_chart(fig_top_tracks, use_container_width=True)
                
                # Mostrar lista das top músicas
                st.write("**Top 10 Artistas (Melhor Música):**")
                display_tracks = top_tracks.copy()
                display_tracks.columns = ['Artista', 'Música', 'Popularidade']
                display_tracks.index = range(1, len(display_tracks) + 1)
                st.dataframe(display_tracks, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao processar dados dos artistas: {str(e)}")
        else:
            st.warning("Nenhum dado encontrado para o filtro selecionado.")
    
    with col2:
        
        # Usar função otimizada com cache
        with st.spinner('Calculando top artistas...'):
            top_artists_by_top3 = calculate_top_artists(selected_genre)
        
        if len(top_artists_by_top3) > 0:
            try:
                fig_pop_artists = px.bar(
                    top_artists_by_top3,
                    x='avg_top3_popularity',
                    y='artist_name',
                    orientation='h',
                    title="Top 10 Artistas (Média das 3 Melhores Músicas)",
                    labels={'avg_top3_popularity': 'Média Top 3 Músicas', 'artist_name': 'Artista'},
                    hover_data=['best_song', 'best_song_popularity', 'total_songs'],
                    color='avg_top3_popularity',
                    color_continuous_scale='Blues'
                )
                fig_pop_artists.update_layout(height=400)
                st.plotly_chart(fig_pop_artists, use_container_width=True)
                
                # Mostrar detalhes dos artistas
                st.write("**Detalhes dos Top Artistas:**")
                details_display = top_artists_by_top3[['artist_name', 'best_song', 'avg_top3_popularity', 'total_songs']].copy()
                details_display.columns = ['Artista', 'Melhor Música', 'Média Top 3', 'Total de Músicas']
                details_display['Média Top 3'] = details_display['Média Top 3'].round(1)
                details_display.index = range(1, len(details_display) + 1)
                st.dataframe(details_display, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao exibir ranking de artistas: {str(e)}")
        else:
            st.warning("Não há dados suficientes para calcular ranking de artistas.")
    
    # 5. Análise de Correlação: Energia vs Volume
    st.subheader("🔍 Análise Interativa: Energia vs Volume")
    
    st.write("**Relação entre Energia Musical e Volume (Loudness)**")
    st.write("Esta análise mostra a correlação linear entre a energia e o volume das músicas.")
    
    fig_scatter = px.scatter(
        filtered_data.sample(min(1500, len(filtered_data))),  # Amostra para performance
        x='energy',
        y='loudness',
        color='popularity',
        title="Energia vs Volume (dB)",
        labels={
            'energy': 'Energia Musical (0.0 - 1.0)',
            'loudness': 'Volume em Decibéis (dB)',
            'popularity': 'Popularidade'
        },
        hover_data=['artist_name', 'track_name', 'tempo'],
        opacity=0.6,
        color_continuous_scale='Plasma'
    )
    
    fig_scatter.update_layout(
        height=500,
        xaxis_title="⚡ Energia Musical",
        yaxis_title="🔊 Volume (dB)"
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Insight da correlação
    correlation = filtered_data['energy'].corr(filtered_data['loudness'])
    if abs(correlation) > 0.3:
        correlation_text = "forte" if abs(correlation) > 0.5 else "moderada"
        direction = "positiva" if correlation > 0 else "negativa"
    else:
        correlation_text = "fraca"
        direction = "positiva" if correlation > 0 else "negativa"
    
    st.info(f"💡 **Insight**: A correlação entre energia e volume é {correlation_text} e {direction} (r = {correlation:.3f}). " +
            f"{'Músicas mais energéticas tendem a ser mais altas!' if correlation > 0.4 else 'A relação entre energia e volume é interessante de observar.'}")
    
    # Estatísticas descritivas
    st.header("📋 Estatísticas Descritivas")
    
    numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
    st.dataframe(filtered_data[numeric_columns].describe(), use_container_width=True)

if __name__ == "__main__":
    main()