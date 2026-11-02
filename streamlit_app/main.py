"""
CropCare AI - Main Application Entry Point
Streamlit multi-page application with proper navigation.
"""

import sys
from pathlib import Path

# Add project root to Python path for src imports
# This is required for Streamlit Cloud deployment where the working directory
# may not include the repository root in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils.constants import PAGE_CONFIG, PAGES
from src.disease_info.translations import get_translator, set_language, get_language
from streamlit_app.components.sidebar import render_sidebar
from src.model.inference import get_inference_engine


def main():
    """Main application entry point with st.navigation()."""
    # Configure page (MUST be first Streamlit command)
    st.set_page_config(**PAGE_CONFIG)

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["current_page"] = "Home"
        st.session_state["language"] = "en"

    # Pre-load model in background for faster first prediction
    if "model_loaded" not in st.session_state:
        with st.spinner("Loading AI model..."):
            try:
                get_inference_engine()
                st.session_state["model_loaded"] = True
            except Exception as e:
                st.session_state["model_loaded"] = False
                st.session_state["model_error"] = str(e)

    # Apply custom CSS
    apply_custom_css()

    # Create page objects for navigation
    pages = create_page_objects()

    # Set up navigation
    pg = st.navigation(pages)
    pg.run()


def create_page_objects():
    """Create st.Page objects for all application pages."""
    from pathlib import Path
    import streamlit as st

    pages_dir = Path(__file__).parent / "pages"

    page_configs = [
        {"name": "Home", "icon": "🏠", "file": "1_Home.py"},
        {"name": "Disease Detection", "icon": "🔬", "file": "2_Disease_Detection.py"},
        {"name": "Prediction History", "icon": "📊", "file": "3_Prediction_History.py"},
        {"name": "Disease Guide", "icon": "📚", "file": "4_Disease_Guide.py"},
        {"name": "Model Information", "icon": "🤖", "file": "5_Model_Information.py"},
    ]

    page_objects = []
    missing_files = []
    for config in page_configs:
        page_path = pages_dir / config["file"]
        page_path_resolved = page_path.resolve()

        # Verify file exists
        if not page_path_resolved.exists():
            missing_files.append(str(page_path_resolved))
            continue

        # Use path relative to entry point directory (streamlit_app/)
        # Streamlit 1.38.0 requires page paths relative to the entry point file
        relative_page_path = page_path.relative_to(Path(__file__).parent)

        page_objects.append(
            st.Page(
                page=str(relative_page_path),
                title=config["name"],
                icon=config["icon"],
                url_path=config["name"].lower().replace(" ", "-"),
            )
        )

    if missing_files:
        st.error(f"Missing page files: {', '.join(missing_files)}")
        st.stop()

    return page_objects


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown(
        """
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #2E7D32;
            --primary-light: #4CAF50;
            --primary-dark: #1B5E20;
            --secondary-color: #E8F5E9;
            --text-color: #1B5E20;
            --text-light: #666;
            --border-color: #e0e0e0;
            --error-color: #F44336;
            --warning-color: #FF9800;
            --success-color: #4CAF50;
        }

        /* Global styles */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--primary-dark) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            border: none;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 12px rgba(46, 125, 50, 0.4);
            transform: translateY(-1px);
        }

        .stButton > button[kind="secondary"] {
            background: white;
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
        }

        .stButton > button[kind="secondary"]:hover {
            background: var(--secondary-color);
        }

        /* Cards */
        .result-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        }

        /* Metrics */
        .stMetric {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid var(--border-color);
        }

        /* Dataframes */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background: var(--secondary-color);
            border-radius: 8px;
            font-weight: 600;
        }

        /* Sidebar */
        .css-1d391kg {
            background: var(--secondary-color);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--primary-color) !important;
            color: white !important;
        }

        /* File uploader */
        .stFileUploader > div {
            border: 2px dashed var(--primary-color);
            border-radius: 12px;
            padding: 2rem;
        }

        .stFileUploader > div:hover {
            background: var(--secondary-color);
        }

        /* Alerts */
        .stAlert {
            border-radius: 12px;
            border: none;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            border-radius: 8px;
        }

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input {
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-dark);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
            .stColumns > div {
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()