import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration & Title
st.set_page_config(page_title="Movie Clustering Dashboard", layout="wide")
st.title("🎬 Movie Clustering Dashboard using K-Means")
st.write("Aapne jo K-Means aur Centroids seekha hai, yeh dashboard usko live data par implement karta hai!")

# 2. Sidebar Controls
st.sidebar.header("K-Means Hyperparameters")
# User yahan se K ki value change kar sakta hai
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=3, step=1)

# 3. Data Loading Function (with Caching for Speed)
@st.cache_data
def load_and_process_data():
    # Files load karein (Ensure these files are in the same folder or update paths)
    movies = pd.read_csv("movie.csv")
    ratings = pd.read_csv("rating.csv")
    
    # Features nikalna: Average Rating aur Total Ratings count per movie
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        total_ratings=('rating', 'count')
    ).reset_index()
    
    # Movies metadata ke sath merge karna
    full_data = pd.merge(movie_stats, movies, on='movieId')
    return full_data

# Data load ho raha hai
try:
    df = load_and_process_data()
    
    # 4. Feature Selection & Scaling
    # K-Means scale-sensitive hota hai, isliye StandardScaler zaroori hai
    features = ['avg_rating', 'total_ratings']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Model Training (K-Means)
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Centroids nikalna (Scaled back to original values for plotting)
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    # 6. Layout UI Columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Clusters Visualization (K = {k_value})")
        
        # Matplotlib Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Har cluster ke points plot karna
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#33FFF0', '#A833FF', '#FF33A8', '#FFA500']
        for i in range(k_value):
            cluster_data = df[df['cluster'] == i]
            ax.scatter(cluster_data['avg_rating'], cluster_data['total_ratings'], 
                       label=f'Cluster {i}', alpha=0.6, c=colors[i], edgecolors='none', s=30)
            
        # Centroids ko 'X' mark ke sath bold plot karna
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids', edgecolors='white')
        
        ax.set_xlabel('Average Rating')
        ax.set_ylabel('Total Number of Ratings')
        ax.set_title('Movies Grouped by Ratings & Popularity')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Display plot in Streamlit
        st.pyplot(fig)

    with col2:
        st.subheader("💡 Centroids Data")
        st.write("Centroid har cluster ka 'Center Point' hota hai:")
        
        # Centroids dataframe tabular view ke liye
        centroid_df = pd.DataFrame(centroids, columns=['Avg Rating (Center)', 'Total Ratings (Center)'])
        centroid_df.index.name = 'Cluster ID'
        st.dataframe(centroid_df.style.format("{:.2f}"))
        
        st.info("💡 **Insight:** Jis cluster ka 'Total Ratings' center high hoga, woh highly popular blockbusters hain!")

    # 7. Cluster Explorers (Show actual data)
    st.markdown("---")
    st.subheader("🔍 Explore Movies Inside Clusters")
    selected_cluster = st.selectbox("Select a Cluster to view movies:", options=range(k_value))
    
    cluster_movies = df[df['cluster'] == selected_cluster][['title', 'genres', 'avg_rating', 'total_ratings']]
    st.write(f"Total movies in Cluster {selected_cluster}: **{len(cluster_movies)}**")
    st.dataframe(cluster_movies.head(50), use_container_width=True)

except FileNotFoundError:
    st.error("❌ 'movies.csv' ya 'ratings.csv' nahi mili! Please check karein ke files sahi folder mein hain.")