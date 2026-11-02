"""
CropCare AI - History Table Component
Searchable, filterable prediction history with actions.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.database.crud import (
    get_predictions,
    delete_prediction,
    clear_all_predictions,
    get_prediction_stats,
    search_predictions,
)
from src.database.models import Prediction
from src.disease_info.translations import get_translator


def render_history_table():
    """Render the complete prediction history page."""
    translator = get_translator()

    st.markdown(f"### 📊 {translator.t('history_title')}")
    st.markdown(f"*{translator.t('history_subtitle')}*")

    # Filters
    render_filters()

    st.divider()

    # Get filtered predictions
    predictions, total = get_filtered_predictions()

    # Summary
    render_summary(total, predictions)

    st.divider()

    # Table
    if predictions:
        render_predictions_table(predictions)
    else:
        st.info(translator.t("no_predictions"))

    # Bulk actions
    if predictions:
        st.divider()
        render_bulk_actions()


def render_filters():
    """Render filter controls."""
    translator = get_translator()

    with st.expander(f"🔍 {translator.t('search_placeholder')}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            search_query = st.text_input(
                translator.t("search_placeholder"),
                key="history_search",
                placeholder=translator.t("search_placeholder"),
            )

        with col2:
            crop_filter = st.selectbox(
                translator.t("filter_crop"),
                options=["", "Tomato", "Potato", "Pepper"],
                format_func=lambda x: translator.t("filter_all") if x == "" else translator.tc(x),
                key="history_crop_filter",
            )

        with col3:
            status_filter = st.selectbox(
                translator.t("filter_status"),
                options=["", "Healthy", "Diseased"],
                format_func=lambda x: translator.t("filter_all") if x == "" else translator.t(x),
                key="history_status_filter",
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            date_from = st.date_input(
                translator.t("date_from"),
                value=None,
                key="history_date_from",
            )

        with col5:
            date_to = st.date_input(
                translator.t("date_to"),
                value=None,
                key="history_date_to",
            )

        with col6:
            if st.button(translator.t("clear_filters"), key="clear_history_filters"):
                for key in ["history_search", "history_crop_filter", "history_status_filter",
                           "history_date_from", "history_date_to"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Store filters in session state
    st.session_state["history_filters"] = {
        "search": search_query,
        "crop": crop_filter,
        "status": status_filter,
        "date_from": date_from,
        "date_to": date_to,
    }


def get_filtered_predictions() -> tuple:
    """Get predictions based on current filters."""
    filters = st.session_state.get("history_filters", {})

    search = filters.get("search", "")
    crop = filters.get("crop", "")
    status = filters.get("status", "")
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    # Convert dates to datetime
    start_dt = None
    end_dt = None
    if date_from:
        start_dt = datetime.combine(date_from, datetime.min.time())
    if date_to:
        end_dt = datetime.combine(date_to, datetime.max.time())

    if search:
        predictions = search_predictions(search, limit=500)
        # Apply additional filters manually
        filtered = []
        for p in predictions:
            if crop and p.crop != crop:
                continue
            if status and p.health_status != status:
                continue
            if start_dt and p.timestamp < start_dt:
                continue
            if end_dt and p.timestamp > end_dt:
                continue
            filtered.append(p)
        return filtered, len(filtered)
    else:
        return get_predictions(
            limit=500,
            crop=crop if crop else None,
            health_status=status if status else None,
            start_date=start_dt,
            end_date=end_dt,
        )


def render_summary(total: int, predictions: list = None):
    """Render summary statistics."""
    translator = get_translator()

    col1, col2, col3, col4 = st.columns(4)

    # If predictions provided, calculate stats from them (filtered)
    if predictions is not None:
        avg_conf = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
        healthy = sum(1 for p in predictions if p.health_status == "Healthy")
        diseased = sum(1 for p in predictions if p.health_status == "Diseased")
    else:
        # Fallback to global stats
        stats = get_prediction_stats()
        avg_conf = stats.get('average_confidence', 0)
        healthy = stats.get("by_health_status", {}).get("Healthy", 0)
        diseased = stats.get("by_health_status", {}).get("Diseased", 0)

    with col1:
        st.metric(translator.t("total_predictions"), total)
    with col2:
        st.metric(translator.t("average_confidence"), f"{avg_conf*100:.1f}%")
    with col3:
        st.metric("✅ Healthy", healthy)
    with col4:
        st.metric("⚠️ Diseased", diseased)


def render_predictions_table(predictions: List[Prediction]):
    """Render predictions as interactive table."""
    translator = get_translator()

    # Convert to DataFrame
    data = []
    for p in predictions:
        data.append({
            "ID": p.id,
            "Date": p.timestamp.strftime("%Y-%m-%d %H:%M") if p.timestamp else "N/A",
            "Crop": translator.tc(p.crop),
            "Disease": translator.td(p.disease),
            "Confidence": f"{p.confidence*100:.1f}%",
            "Status": "✅ " + translator.t("healthy") if p.health_status == "Healthy" else "⚠️ " + translator.t("diseased"),
            "Language": p.language.upper(),
        })

    df = pd.DataFrame(data)

    # Display with selection
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Date": st.column_config.DatetimeColumn("Date", width="medium"),
            "Crop": st.column_config.TextColumn("Crop", width="small"),
            "Disease": st.column_config.TextColumn("Disease", width="large"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Language": st.column_config.TextColumn("Lang", width="small"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle row selection
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_pred = predictions[selected_idx]
        render_prediction_detail(selected_pred)


def render_prediction_detail(prediction: Prediction):
    """Render detailed view of a prediction."""
    translator = get_translator()

    with st.expander(f"📋 {translator.t('view_details')} - ID {prediction.id}", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**{translator.t('crop_detected')}:** {translator.tc(prediction.crop)}")
            st.markdown(f"**{translator.t('disease_detected')}:** {translator.td(prediction.disease)}")
            st.markdown(f"**{translator.t('confidence')}:** {prediction.confidence*100:.1f}%")
            st.markdown(f"**{translator.t('health_status')}:** {translator.t(prediction.health_status.lower())}")
            st.markdown(f"**{translator.t('language')}:** {prediction.language.upper()}")
            st.markdown(f"**{translator.t('timestamp')}:** {prediction.timestamp}")

            if prediction.notes:
                st.markdown(f"**Notes:** {prediction.notes}")

        with col2:
            # Action buttons
            if st.button(translator.t("delete_record"), key=f"delete_{prediction.id}", type="secondary"):
                if delete_prediction(prediction.id):
                    st.success(f"Deleted prediction {prediction.id}")
                    st.rerun()
                else:
                    st.error("Failed to delete")

            if prediction.image_path:
                st.markdown(f"**Image:** {prediction.image_path}")
            if prediction.gradcam_path:
                st.markdown(f"**Grad-CAM:** {prediction.gradcam_path}")


def render_bulk_actions():
    """Render bulk action buttons."""
    translator = get_translator()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"🗑️ {translator.t('clear_history')}", type="secondary", key="bulk_clear"):
            st.session_state["show_clear_confirm"] = True

    with col2:
        # Export button
        if st.button("📥 Export CSV", type="secondary", key="export_history"):
            export_history_csv()

    # Confirmation dialog for clear all
    if st.session_state.get("show_clear_confirm"):
        with st.container():
            st.error(f"⚠️ {translator.t('confirm_clear')}")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button(f"✅ {translator.t('yes')}, {translator.t('clear_history')}", type="primary", key="confirm_clear_yes"):
                    count = clear_all_predictions(confirm=True)
                    st.success(f"Cleared {count} predictions")
                    st.session_state["show_clear_confirm"] = False
                    st.rerun()
            with col_no:
                if st.button(f"❌ {translator.t('no')}", key="confirm_clear_no"):
                    st.session_state["show_clear_confirm"] = False
                    st.rerun()


def export_history_csv():
    """Export prediction history to CSV."""
    predictions, _ = get_predictions(limit=10000)

    if not predictions:
        st.warning("No data to export")
        return

    data = []
    for p in predictions:
        data.append({
            "ID": p.id,
            "Timestamp": p.timestamp.isoformat() if p.timestamp else "",
            "Crop": p.crop,
            "Disease": p.disease,
            "Confidence": p.confidence,
            "Health_Status": p.health_status,
            "Image_Path": p.image_path or "",
            "GradCAM_Path": p.gradcam_path or "",
            "Language": p.language,
            "Notes": p.notes or "",
        })

    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"cropcare_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_history_sidebar():
    """Render compact history in sidebar."""
    translator = get_translator()

    from src.database.crud import get_recent_predictions

    recent = get_recent_predictions(limit=5)

    if recent:
        st.markdown("#### 📜 Recent Predictions")
        for pred in recent:
            status_icon = "✅" if pred.health_status == "Healthy" else "⚠️"
            st.markdown(
                f"{status_icon} **{translator.tc(pred.crop)}** - {translator.td(pred.disease)} "
                f"({pred.confidence*100:.0f}%)"
            )
    else:
        st.markdown("*No recent predictions*")


if __name__ == "__main__":
    pass