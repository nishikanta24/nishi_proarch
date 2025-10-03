#!/usr/bin/env python3
"""
Streamlit Frontend for AI Cost & Insights Copilot
A simple web interface for cost analytics and Q&A
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30

# Page configuration
st.set_page_config(
    page_title="AI Cost & Insights Copilot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #ffffff;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #4a5568;
    }
    .chat-user {
        background-color: #1f77b4;
        color: white;
        margin-left: auto;
        border: none;
    }
    .chat-assistant {
        background-color: #2d3748;
        color: #ffffff;
        border-left: 4px solid #1f77b4;
        border: 1px solid #4a5568;
    }
    .source-highlight {
        background-color: #2d3748;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    /* Dark theme improvements */
    .stApp {
        background-color: #1a202c;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
    .stTextInput input {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    .stTextArea textarea {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    .stButton button {
        background-color: #1f77b4;
        color: #ffffff;
        border: none;
    }
    .stButton button:hover {
        background-color: #1a5d8a;
    }
</style>
""", unsafe_allow_html=True)

def api_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=API_TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=API_TIMEOUT)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def display_kpi_metrics(kpi_data):
    """Display KPI metrics in a nice layout"""
    if not kpi_data or 'kpis' not in kpi_data:
        st.error("No KPI data available")
        return

    kpis = kpi_data['kpis']
    month = kpi_data.get('month', 'Unknown')

    st.subheader(f"📊 Cost Analytics Dashboard - {month}")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_cost = kpis.get('total_cost', 0)
        st.metric("Total Cost", f"${total_cost:,.2f}")

    with col2:
        prev_cost = kpis.get('previous_month_cost', 0)
        if prev_cost > 0:
            change_pct = ((total_cost - prev_cost) / prev_cost) * 100
            st.metric("Month-over-Month", f"{change_pct:+.1f}%", delta=f"${total_cost - prev_cost:,.2f}")
        else:
            st.metric("Month-over-Month", "N/A")

    with col3:
        top_services = kpis.get('top_services', [])
        st.metric("Top Services", len(top_services))

    with col4:
        resources = kpis.get('resource_count', 0)
        st.metric("Total Resources", resources)

    # Charts section
    st.subheader("📈 Cost Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Top services by cost
        if top_services:
            services_df = pd.DataFrame(top_services)
            fig = px.bar(
                services_df.head(10),
                x='service',
                y='cost',
                title="Top 10 Services by Cost",
                labels={'cost': 'Cost ($)', 'service': 'Service'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cost by region (if available)
        regions = kpis.get('cost_by_region', [])
        if regions:
            regions_df = pd.DataFrame(regions)
            fig = px.pie(
                regions_df.head(10),
                values='cost',
                names='region',
                title="Cost Distribution by Region"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Additional insights
    st.subheader("🔍 Key Insights")

    insights = []

    # Cost anomalies
    anomalies = kpis.get('anomalies', [])
    if anomalies:
        insights.append(f"⚠️ Found {len(anomalies)} cost anomalies detected")

    # Tagging compliance
    tagging = kpis.get('tagging_compliance', {})
    if tagging:
        compliance_rate = tagging.get('compliance_rate', 0)
        insights.append(f"🏷️ Resource tagging compliance: {compliance_rate:.1f}%")

    # Top cost drivers
    if top_services:
        top_service = top_services[0] if top_services else {}
        service_name = top_service.get('service', 'Unknown')
        service_cost = top_service.get('cost', 0)
        insights.append(f"💰 Largest cost driver: {service_name} (${service_cost:,.2f})")

    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.info("No additional insights available for this month.")

def display_chat_message(message, is_user=False):
    """Display a chat message with proper styling"""
    css_class = "chat-user" if is_user else "chat-assistant"

    col1, col2 = st.columns([1, 4])
    with col2 if is_user else col1:
        st.markdown(f"""
        <div class="chat-message {css_class}">
            {message}
        </div>
        """, unsafe_allow_html=True)

def display_qa_response(response_data):
    """Display Q&A response with sources and suggestions"""
    if not response_data:
        st.error("No response data available")
        return

    # Main answer
    answer = response_data.get('answer', 'No answer provided')
    display_chat_message(answer, is_user=False)

    # Sources (if available)
    sources = response_data.get('sources', [])
    if sources:
        with st.expander("📚 Sources & Context", expanded=False):
            for i, source in enumerate(sources[:3]):  # Limit to top 3
                st.markdown(f"""
                <div class="source-highlight">
                    <strong>Source {i+1}:</strong><br>
                    {source[:300]}{'...' if len(source) > 300 else ''}
                </div>
                """, unsafe_allow_html=True)

    # Data table (if available)
    data_table = response_data.get('data_table')
    if data_table:
        with st.expander("📊 Data Table", expanded=False):
            try:
                # Try to parse and display as DataFrame
                if isinstance(data_table, str):
                    # Assume it's a markdown table or JSON
                    st.code(data_table, language='markdown')
                elif isinstance(data_table, dict):
                    df = pd.DataFrame(data_table)
                    st.dataframe(df)
                else:
                    st.code(str(data_table), language='json')
            except Exception as e:
                st.code(str(data_table), language='text')

    # Suggestions (if available)
    suggestions = response_data.get('suggestions', [])
    if suggestions:
        with st.expander("💡 Recommendations", expanded=False):
            for suggestion in suggestions:
                st.success(f"• {suggestion}")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">💰 AI Cost & Insights Copilot</h1>', unsafe_allow_html=True)
    st.markdown("*Your intelligent cloud cost analysis assistant*")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")

        # API Status
        st.subheader("API Status")
        if st.button("🔍 Check API Health"):
            with st.spinner("Checking API..."):
                health = api_request("/api/v1/health")
                if health:
                    st.success("✅ API is healthy")
                    st.json(health)
                else:
                    st.error("❌ API is not responding")

        st.divider()

        # Month selector for KPIs
        st.subheader("📅 KPI Month")
        selected_month = st.text_input(
            "Month (YYYY-MM)",
            value=datetime.now().strftime("%Y-%m"),
            help="Enter month in YYYY-MM format"
        )

        if st.button("📊 Load KPIs", type="primary"):
            with st.spinner("Loading KPI data..."):
                kpi_data = api_request(f"/api/v1/kpi?month={selected_month}")
                if kpi_data:
                    st.session_state.kpi_data = kpi_data
                    st.success(f"✅ Loaded KPI data for {selected_month}")
                else:
                    st.error("❌ Failed to load KPI data")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Ask Questions", "ℹ️ About"])

    # Dashboard Tab
    with tab1:
        st.header("Cost Analytics Dashboard")

        # Load KPI data if available
        if 'kpi_data' in st.session_state:
            display_kpi_metrics(st.session_state.kpi_data)
        else:
            st.info("👆 Select a month and click 'Load KPIs' to view the dashboard")

            # Quick demo button
            if st.button("🚀 Load Latest KPIs", type="primary"):
                with st.spinner("Loading latest KPI data..."):
                    kpi_data = api_request("/api/v1/kpi")
                    if kpi_data:
                        st.session_state.kpi_data = kpi_data
                        st.rerun()
                    else:
                        st.error("❌ Failed to load KPI data")

    # Chat Tab
    with tab2:
        st.header("Ask Questions")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(message['content'], message['is_user'])

        # Chat input
        with st.form(key='chat_form', clear_on_submit=True):
            user_question = st.text_area(
                "Ask a question about your cloud costs:",
                placeholder="e.g., 'Why did my costs increase last month?' or 'Which resources are underutilized?'",
                height=100
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("🚀 Ask", type="primary", use_container_width=True)

        if submit_button and user_question.strip():
            # Add user message to history
            st.session_state.chat_history.append({
                'content': user_question,
                'is_user': True
            })

            # Show typing indicator
            with st.spinner("🤔 Thinking..."):
                # Make API call
                qa_response = api_request("/api/v1/ask", method="POST", data={"question": user_question})

            if qa_response:
                # Add assistant response to history
                response_content = qa_response.get('answer', 'No answer provided')
                st.session_state.chat_history.append({
                    'content': response_content,
                    'is_user': False,
                    'full_response': qa_response
                })

                # Display the response with additional details
                display_qa_response(qa_response)
            else:
                error_msg = "Sorry, I couldn't get an answer right now. Please try again."
                st.session_state.chat_history.append({
                    'content': error_msg,
                    'is_user': False
                })
                st.error(error_msg)

            # Rerun to update chat display
            st.rerun()

    # About Tab
    with tab3:
        st.header("About")

        st.markdown("""
        ## AI Cost & Insights Copilot

        This application helps FinOps analysts analyze cloud costs through:

        ### 🎯 **Key Features**
        - **Cost Analytics Dashboard**: View KPIs, trends, and cost breakdowns
        - **Natural Language Q&A**: Ask questions in plain English about your costs
        - **AI-Powered Recommendations**: Get actionable insights for cost optimization

        ### 🔧 **Technology Stack**
        - **Backend**: FastAPI with Python
        - **AI**: RAG (Retrieval-Augmented Generation) with vector search
        - **Database**: SQLite with cost and resource data
        - **Frontend**: Streamlit (this interface)

        ### 📊 **Data Sources**
        - Monthly billing data (CSV)
        - Resource metadata and tags
        - Cost optimization reference documents

        ### 🚀 **Getting Started**
        1. The API server should be running on `http://localhost:8000`
        2. Use the Dashboard tab to view cost analytics
        3. Use the Ask Questions tab to query your data

        ### 💡 **Example Questions**
        - "What was my total spend last month?"
        - "Why did costs increase compared to previous month?"
        - "Which resources look idle?"
        - "Show me cost breakdown by service"
        """)

        # API endpoints info
        with st.expander("🔌 API Endpoints"):
            st.code("""
GET  /api/v1/kpi?month=YYYY-MM          # Get KPIs for a month
POST /api/v1/ask                         # Ask questions (JSON body)
POST /api/v1/recommendations              # Get cost optimization recommendations
GET  /api/v1/health                      # Health check
            """)

if __name__ == "__main__":
    main()
