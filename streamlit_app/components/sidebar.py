"""
CropCare AI - Sidebar Component
Language switcher and quick stats for the sidebar.
Uses native Streamlit components only.
"""

import streamlit as st
from src.disease_info.translations import get_translator, set_language, get_language


def render_sidebar():
    """Render sidebar content (language selector, stats, footer)."""
    with st.sidebar:
        st.divider()

        # Language selector
        render_language_selector()

        st.divider()

        # Quick stats
        render_quick_stats()

        st.divider()

        # Footer
        render_sidebar_footer()


def render_language_selector():
    """Render language selection dropdown."""
    translator = get_translator()
    current_lang = translator.get_lang()

    lang_options = {
        "en": "🇺🇸 English",
        "hi": "🇮🇳 हिंदी",
    }

    selected = st.selectbox(
        "🌐 Language / भाषा",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(current_lang),
        key="language_selector",
    )

    if selected != current_lang:
        set_language(selected)
        st.rerun()


def render_quick_stats():
    """Render quick statistics in sidebar."""
    from src.database.crud import get_prediction_stats

    try:
        stats = get_prediction_stats()

        st.markdown("### 📈 Quick Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", stats.get("total_predictions", 0))
            st.metric("This Week", stats.get("recent_predictions_7d", 0))
        with col2:
            avg_conf = stats.get("average_confidence", 0)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

            # Health status breakdown
            healthy = stats.get("by_health_status", {}).get("Healthy", 0)
            diseased = stats.get("by_health_status", {}).get("Diseased", 0)
            st.metric("Healthy / Diseased", f"{healthy} / {diseased}")

    except Exception:
        # Silently fail if database not initialized
        pass


def render_sidebar_footer():
    """Render sidebar footer with links."""
    st.markdown("---")
    st.caption("CropCare AI v1.0.0")
    st.caption("B.Tech Project")

    # Links
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📖 Docs", "https://github.com/yourusername/CropCare-AI")
    with col2:
        st.link_button("🐛 Issues", "https://github.com/yourusername/CropCare-AI/issues")


if __name__ == "__main__":
    pass