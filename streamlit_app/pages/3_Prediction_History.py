"""
CropCare AI - Prediction History Page
View, search, filter, and manage prediction history with charts and statistics.
"""

import sys
from pathlib import Path

# Add project root to Python path for src imports
# Required for Streamlit Cloud deployment
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.disease_info.translations import get_translator, get_ui_translation
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.history_table import render_history_table, get_filtered_predictions
from src.database.crud import get_prediction_stats


def main():
    # Render sidebar (page config is set in main.py via st.navigation)
    render_sidebar()

    # Get translator
    translator = get_translator()

    # Page title
    st.markdown(f"# 📊 {translator.t('history_title')}")
    st.markdown(f"*{translator.t('history_subtitle')}*")

    st.divider()

    # Get filtered predictions for charts
    predictions, total = get_filtered_predictions()

    # Dashboard charts section
    if predictions:
        render_dashboard_charts(predictions, translator)

    st.divider()

    # Render history table with all functionality
    render_history_table()

    # Disclaimer
    st.divider()
    st.warning(get_ui_translation("disclaimer", translator.get_lang()))


def render_dashboard_charts(predictions, translator):
    """Render dashboard charts for prediction analytics."""
    st.markdown(f"## 📈 {translator.t('analytics_title')}")

    # Convert to DataFrame
    df = pd.DataFrame([{
        "timestamp": p.timestamp,
        "crop": p.crop,
        "disease": p.disease,
        "confidence": p.confidence,
        "health_status": p.health_status,
    } for p in predictions])

    if df.empty:
        return

    # Row 1: Health status pie chart and confidence distribution
    col1, col2 = st.columns(2)

    with col1:
        # Health status pie chart
        health_counts = df['health_status'].value_counts().reset_index()
        health_counts.columns = ['Health Status', 'Count']
        fig_pie = px.pie(
            health_counts,
            values='Count',
            names='Health Status',
            title=translator.t('health_distribution'),
            color='Health Status',
            color_discrete_map={'Healthy': '#4CAF50', 'Diseased': '#F44336'},
            hole=0.4,
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Confidence distribution histogram
        fig_hist = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title=translator.t('confidence_distribution'),
            labels={'confidence': 'Confidence', 'count': 'Number of Predictions'},
            color_discrete_sequence=['#2E7D32'],
        )
        fig_hist.update_layout(
            xaxis_title="Confidence",
            yaxis_title="Count",
            height=350,
            bargap=0.1,
        )
        fig_hist.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_hist, use_container_width=True)

    # Row 2: Predictions by crop and disease
    col3, col4 = st.columns(2)

    with col3:
        # Predictions by crop
        crop_counts = df['crop'].value_counts().reset_index()
        crop_counts.columns = ['Crop', 'Count']
        fig_crop = px.bar(
            crop_counts,
            x='Crop',
            y='Count',
            title=translator.t('predictions_by_crop'),
            color='Crop',
            color_discrete_sequence=px.colors.qualitative.Set2,
            text='Count',
        )
        fig_crop.update_traces(textposition='outside')
        fig_crop.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_crop, use_container_width=True)

    with col4:
        # Top diseases
        disease_counts = df['disease'].value_counts().head(10).reset_index()
        disease_counts.columns = ['Disease', 'Count']
        fig_disease = px.bar(
            disease_counts,
            x='Count',
            y='Disease',
            orientation='h',
            title=translator.t('top_diseases'),
            color='Count',
            color_continuous_scale='Reds',
            text='Count',
        )
        fig_disease.update_traces(textposition='outside')
        fig_disease.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_disease, use_container_width=True)

    # Row 3: Predictions over time (if enough data)
    if len(df) > 1:
        st.markdown(f"### 📅 {translator.t('predictions_over_time')}")

        # Group by date
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby(['date', 'health_status']).size().reset_index(name='Count')

        fig_time = px.line(
            daily_counts,
            x='date',
            y='Count',
            color='health_status',
            title=translator.t('daily_predictions'),
            labels={'date': 'Date', 'Count': 'Number of Predictions'},
            color_discrete_map={'Healthy': '#4CAF50', 'Diseased': '#F44336'},
            markers=True,
        )
        fig_time.update_layout(height=350, xaxis_title="Date", yaxis_title="Predictions")
        st.plotly_chart(fig_time, use_container_width=True)

    # Summary metrics row
    st.markdown(f"### 📋 {translator.t('summary_metrics')}")
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_conf = df['confidence'].mean()
    healthy_count = (df['health_status'] == 'Healthy').sum()
    diseased_count = (df['health_status'] == 'Diseased').sum()
    unique_crops = df['crop'].nunique()
    unique_diseases = df['disease'].nunique()

    with col1:
        st.metric(translator.t('total_predictions'), len(df))
    with col2:
        st.metric(translator.t('average_confidence'), f"{avg_conf*100:.1f}%")
    with col3:
        st.metric("✅ Healthy", healthy_count)
    with col4:
        st.metric("⚠️ Diseased", diseased_count)
    with col5:
        st.metric(translator.t('diseases_detected'), unique_diseases)


if __name__ == "__main__":
    main()