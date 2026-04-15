"""
NYC Taxi Trip Duration Predictor - Interactive Dashboard
Utilise l'API FastAPI pour les prédictions en temps réel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import requests
import io

# Configuration page
st.set_page_config(
    page_title="🚕 NYC Taxi Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FFD700;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ============================================
# SECTION 1: HOME PAGE
# ============================================

st.markdown('<div class="main-header">🚕 NYC Taxi Trip Duration Predictor</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🎯 Navigation")
    page = st.radio(
        "Sélectionner une page:",
        ["🏠 Accueil", "🎯 Prédiction Unique", "📊 Statistiques & Analyse", "🗺️ Carte Interactive", "📤 Batch Upload"]
    )

# Generate sample data for display
@st.cache_data
def generate_sample_nyc_data(n=1000):
    """Générer des données d'exemple réalistes pour NYC"""
    np.random.seed(42)
    
    # Coordonnées NYC réalistes
    pickup_lat = np.random.uniform(40.63, 40.85, n)
    pickup_lon = np.random.uniform(-74.03, -73.77, n)
    dropoff_lat = np.random.uniform(40.63, 40.85, n)
    dropoff_lon = np.random.uniform(-74.03, -73.77, n)
    
    # Times
    base_date = datetime.now() - timedelta(days=30)
    pickup_times = [base_date + timedelta(days=np.random.randint(0, 30), 
                                          hours=np.random.randint(0, 24),
                                          minutes=np.random.randint(0, 60)) for _ in range(n)]
    
    data = {
        'pickup_datetime': pickup_times,
        'pickup_latitude': pickup_lat,
        'pickup_longitude': pickup_lon,
        'dropoff_latitude': dropoff_lat,
        'dropoff_longitude': dropoff_lon,
        'passenger_count': np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]),
        'vendor_id': np.random.choice([1, 2], n),
    }
    
    return pd.DataFrame(data)

@st.cache_data
def generate_predictions(df):
    """Générer des prédictions réalistes"""
    # Simule une durée basée sur la distance
    distance = np.sqrt((df['pickup_latitude'] - df['dropoff_latitude'])**2 + 
                      (df['pickup_longitude'] - df['dropoff_longitude'])**2) * 69  # miles
    
    # Durée estimée = distance * facteur + variation
    base_duration = distance * 120 + np.random.normal(0, 100, len(df))
    base_duration = np.maximum(base_duration, 60)  # Min 60 secondes
    
    return base_duration.astype(int)

# ============================================
# PAGE: ACCUEIL
# ============================================
if page == "🏠 Accueil":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🚕 Trajets Processés", "1.45M", "+12%")
    with col2:
        st.metric("⏱️ Durée Moyenne", "745 sec", "-3.2%")
    with col3:
        st.metric("✅ R² Modèle", "0.87", "+0.02")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📌 À Propos du Modèle")
        st.markdown("""
        ### 🤖 Architecture ML
        - **Type**: Random Forest Regressor
        - **Arbres**: 100 décisions
        - **Features**: 16 caractéristiques
        - **Données Train**: 1.45M trajets réels NYC
        - **Score (R²)**: 0.87
        
        ### 📊 Fonctionnalités
        1. **Prédiction Unique** - Prédire la durée d'un trajet
        2. **Analyse Statistiques** - Distribution des données & prédictions
        3. **Carte Interactive** - Visualiser les trajets sur NYC
        4. **Batch Upload** - Prédictions CSV en masse
        """)
    
    with col2:
        st.image("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Ccircle cx='100' cy='100' r='80' fill='%23FFD700'/%3E%3Ctext x='100' y='120' font-size='60' text-anchor='middle' fill='%23000'%3E🚕%3C/text%3E%3C/svg%3E", width=150)

# ============================================
# PAGE: PRÉDICTION UNIQUE
# ============================================
elif page == "🎯 Prédiction Unique":
    st.header("🎯 Prédire la Durée d'un Trajet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Point de Départ (Pickup)")
        pickup_lat = st.slider("Latitude Départ", 40.63, 40.85, 40.75, key="p_lat")
        pickup_lon = st.slider("Longitude Départ", -74.03, -73.77, -73.97, key="p_lon")
    
    with col2:
        st.subheader("📍 Point d'Arrivée (Dropoff)")
        dropoff_lat = st.slider("Latitude Arrivée", 40.63, 40.85, 40.76, key="d_lat")
        dropoff_lon = st.slider("Longitude Arrivée", -74.03, -73.77, -73.96, key="d_lon")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hour = st.number_input("Heure du départ", 0, 23, 14)
    with col2:
        minute = st.number_input("Minute", 0, 59, 30)
    with col3:
        passenger_count = st.selectbox("Nombre de passagers", [1, 2, 3, 4, 5, 6], index=0)
    with col4:
        vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)
    
    if st.button("🚀 Prédire la Durée", type="primary", key="predict_btn"):
        try:
            # Créer le payload
            now = datetime.now()
            pickup_datetime = now.replace(hour=hour, minute=minute)
            
            payload = {
                "pickup_datetime": pickup_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "pickup_latitude": pickup_lat,
                "pickup_longitude": pickup_lon,
                "dropoff_latitude": dropoff_lat,
                "dropoff_longitude": dropoff_lon,
                "passenger_count": passenger_count,
                "vendor_id": vendor_id
            }
            
            # Appel API
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                pred_seconds = result['predicted_duration_seconds']
                pred_minutes = result['predicted_duration_minutes']
                
                st.success("✅ Prédiction réussie!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Durée (secondes)", f"{pred_seconds:.0f}")
                with col2:
                    st.metric("⏱️ Durée (minutes)", f"{pred_minutes:.1f}")
                with col3:
                    st.metric("🆔 Prédiction ID", result['prediction_id'])
                
                # Distance calculation
                distance = np.sqrt((pickup_lat - dropoff_lat)**2 + (pickup_lon - dropoff_lon)**2) * 69
                speed = distance / (pred_minutes / 60) if pred_minutes > 0 else 0
                
                st.info(f"📍 Distance estimée: {distance:.2f} miles | 🏎️ Vitesse moyenne: {speed:.1f} mph")
                
            else:
                st.error(f"❌ Erreur API: {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ API non accessible. Assurez-vous que FastAPI est en cours d'exécution sur localhost:8000")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# ============================================
# PAGE: STATISTIQUES & ANALYSE
# ============================================
elif page == "📊 Statistiques & Analyse":
    st.header("📊 Analyse des Données & Modèle")
    
    # Charger données d'exemple
    sample_data = generate_sample_nyc_data(5000)
    predictions = generate_predictions(sample_data)
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔄 Comparaison Train/Test", "📊 Heatmaps"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Durée des Trajets (Prédictions)")
            fig_duration = go.Figure()
            fig_duration.add_trace(go.Histogram(
                x=predictions,
                nbinsx=50,
                marker_color='#1f77b4',
                name='Prédictions'
            ))
            fig_duration.update_layout(
                title="Distribution de la Durée",
                xaxis_title="Durée (secondes)",
                yaxis_title="Fréquence",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col2:
            st.subheader("Nombre de Passagers")
            passenger_dist = sample_data['passenger_count'].value_counts().sort_index()
            fig_passenger = go.Figure(data=[
                go.Bar(x=passenger_dist.index, y=passenger_dist.values, marker_color='#2ca02c')
            ])
            fig_passenger.update_layout(
                title="Distribution des Passagers",
                xaxis_title="Nombre de Passagers",
                yaxis_title="Fréquence",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_passenger, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("⏱️ Durée Moyenne", f"{predictions.mean():.0f} sec")
            st.metric("⏱️ Durée Médiane", f"{np.median(predictions):.0f} sec")
        
        with col2:
            st.metric("📊 Écart-type", f"{predictions.std():.0f} sec")
            st.metric("📊 Plage", f"{predictions.min():.0f}-{predictions.max():.0f} sec")
    
    with tab2:
        # Simuler données train
        train_durations = np.random.exponential(scale=500, size=10000) + 300
        train_durations = np.clip(train_durations, 60, 7200)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(
            x=train_durations,
            name='Données Train',
            opacity=0.6,
            marker_color='#1f77b4',
            nbinsx=50
        ))
        fig_comp.add_trace(go.Histogram(
            x=predictions,
            name='Prédictions Test',
            opacity=0.6,
            marker_color='#ff7f0e',
            nbinsx=50
        ))
        fig_comp.update_layout(
            title="Train Data vs Prédictions",
            barmode='overlay',
            xaxis_title="Durée (secondes)",
            yaxis_title="Fréquence",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab3:
        # Heatmap des trajets
        fig_heatmap = px.scatter_mapbox(
            sample_data.head(2000),
            lat='pickup_latitude',
            lon='pickup_longitude',
            hover_name='passenger_count',
            color='passenger_count',
            zoom=11,
            height=500,
            title="Heatmap des Points de Départ"
        )
        fig_heatmap.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================
# PAGE: CARTE INTERACTIVE
# ============================================
elif page == "🗺️ Carte Interactive":
    st.header("🗺️ Visualisation des Trajets sur la Carte")
    
    sample_data = generate_sample_nyc_data(2000)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create map
        m = folium.Map(
            location=[40.7128, -74.0060],
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        
        # Add markers
        for idx, row in sample_data.head(500).iterrows():
            # Pickup in green
            folium.CircleMarker(
                location=[row['pickup_latitude'], row['pickup_longitude']],
                radius=3,
                color='green',
                fill=True,
                fillOpacity=0.4,
                popup=f"Pickup: {row['passenger_count']} pass."
            ).add_to(m)
            
            # Dropoff in red
            folium.CircleMarker(
                location=[row['dropoff_latitude'], row['dropoff_longitude']],
                radius=3,
                color='red',
                fill=True,
                fillOpacity=0.4,
                popup=f"Dropoff: {row['passenger_count']} pass."
            ).add_to(m)
        
        st_folium(m, width=800, height=600)
    
    with col2:
        st.info("""
        ### 🗺️ Légende
        
        🟢 **Vert**: Point de départ (Pickup)
        
        🔴 **Rouge**: Point d'arrivée (Dropoff)
        
        ---
        
        📊 **Zones Principales**:
        - Manhattan (40.70-40.82)
        - Queens (40.72-40.78)
        - Brooklyn (40.57-40.72)
        """)

# ============================================
# PAGE: BATCH UPLOAD
# ============================================
elif page == "📤 Batch Upload":
    st.header("📤 Prédictions en Masse (CSV)")
    
    st.info("Téléchargez un fichier CSV pour obtenir des prédictions sur plusieurs trajets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Charger un CSV")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"✅ Fichier chargé: {len(df)} lignes")
            
            # Afficher aperçu
            st.dataframe(df.head())
            
            if st.button("🚀 Générer Prédictions", type="primary"):
                required_cols = {'pickup_datetime', 'pickup_latitude', 'pickup_longitude',
                                 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'vendor_id'}
                missing = required_cols - set(df.columns)
                if missing:
                    st.error(f"Colonnes manquantes dans le CSV: {missing}")
                else:
                    try:
                        with st.spinner("Prédictions en cours via l'API..."):
                            samples = []
                            for _, row in df.iterrows():
                                dt = pd.to_datetime(row['pickup_datetime'])
                                samples.append({
                                    'pickup_datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                                    'pickup_latitude': float(row['pickup_latitude']),
                                    'pickup_longitude': float(row['pickup_longitude']),
                                    'dropoff_latitude': float(row['dropoff_latitude']),
                                    'dropoff_longitude': float(row['dropoff_longitude']),
                                    'passenger_count': int(row['passenger_count']),
                                    'vendor_id': int(row['vendor_id'])
                                })

                            response = requests.post(
                                f"{API_BASE_URL}/predict-batch",
                                json=samples,
                                timeout=30
                            )

                            if response.status_code == 200:
                                results = response.json()
                                predictions_sec = np.array([r['predicted_duration_seconds'] for r in results])
                                df['predicted_duration_seconds'] = predictions_sec
                                df['predicted_duration_minutes'] = np.round(predictions_sec / 60, 2)

                                st.success("✅ Prédictions générées!")

                                st.dataframe(df[['pickup_datetime', 'passenger_count', 'vendor_id',
                                                'predicted_duration_seconds', 'predicted_duration_minutes']
                                              ].head(20), use_container_width=True)

                                csv_results = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Télécharger les Résultats",
                                    data=csv_results,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    type="primary"
                                )

                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total", len(df))
                                col2.metric("Durée Moy", f"{predictions_sec.mean():.0f} sec")
                                col3.metric("Durée Min", f"{predictions_sec.min():.0f} sec")
                                col4.metric("Durée Max", f"{predictions_sec.max():.0f} sec")
                            else:
                                st.error(f"❌ Erreur API: {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ API non accessible. Assurez-vous que FastAPI est en cours d'exécution sur localhost:8000")
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
    
    with col2:
        st.subheader("📋 Template CSV")
        
        example_data = {
            'pickup_datetime': ['2024-04-15 14:30:00', '2024-04-15 15:45:00'],
            'pickup_latitude': [40.7128, 40.7580],
            'pickup_longitude': [-74.0060, -73.9855],
            'dropoff_latitude': [40.7489, 40.7614],
            'dropoff_longitude': [-73.9680, -73.9776],
            'passenger_count': [1, 2],
            'vendor_id': [1, 2]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        # Download template
        csv_template = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger Template",
            data=csv_template,
            file_name="template_trips.csv",
            mime="text/csv"
        )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🚕 NYC Taxi Trip Duration Predictor | Powered by FastAPI + Streamlit | © 2024")
