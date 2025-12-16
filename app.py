import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Peer-Group Anomaly Detection",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 "Cool" Color Scheme Definition ---
# Cyberpunk / High-Contrast Theme
colors = {
    'background': '#0E1117',
    'text': '#FAFAFA',
    'normal': '#00F0FF',   # Neon Cyan
    'outlier': '#FF2B2B',  # Neon Red
    'accent': '#FFD700'    # Gold
}

# Common Chart Template
def apply_theme(fig):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif")
    )
    return fig

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        universe = pd.read_csv('ind_nifty500list.csv')
        universe = universe[['Symbol', 'Company Name']]
        
        watchlist = pd.read_csv('final_project_watchlist_complete.csv')
        
        data = pd.merge(watchlist, universe, on='Symbol', how='left')
        data['Company Name'] = data['Company Name'].fillna(data['Symbol'])
        
        # Logic: -1 is Outlier
        data['Status'] = data['Anomaly_Label'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')
        return data
    except FileNotFoundError:
        st.error("❌ Data files missing. Please upload the CSVs.")
        return pd.DataFrame()

df = load_data()

# --- Sidebar ---
st.sidebar.title("🕵️‍♂️ Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "The Watchlist", "Sector Analysis", "Company Deep Dive", "Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **Model:** Isolation Forest (Unsupervised)")
st.sidebar.caption("📊 **Data:** Nifty 500 (TTM)")
st.sidebar.caption("📅 **Period:** FY 2024-25")

# --- Helper Info ---
feature_cols = ['TATA_Z', 'DSRI_Z', 'AQI_Z', 'P_OCF_Z', 'PEG_Z', 'DuPont_Discrepancy_Z']
feature_names = {
    'TATA_Z': 'Accruals (TATA)', 'DSRI_Z': 'Receivables (DSRI)',
    'AQI_Z': 'Asset Quality (AQI)', 'P_OCF_Z': 'Price/CashFlow',
    'PEG_Z': 'PEG Ratio', 'DuPont_Discrepancy_Z': 'DuPont Discrepancy'
}

# ==========================================
# PAGE 1: EXECUTIVE SUMMARY (FIXED)
# ==========================================
if page == "Executive Summary":
    # 1. Header & Mission Statement
    st.title("🛡️ Peer-Group Anomaly Detection")
    st.markdown("""
    <h3 style='color: #00F0FF; margin-top: -20px;'>
    AI-Driven Forensic Audit of Indian Equities
    </h3>
    """, unsafe_allow_html=True)
    
    st.info("""
    **👋 What is this dashboard?** This tool uses **Unsupervised Machine Learning (Isolation Forests)** to scan the Nifty 500. 
    It identifies companies whose financial accounting ratios (Accruals, Asset Quality, Valuation) 
    **deviate significantly** from their sector peers. 
    
    **🎯 Goal:** Flag potential accounting irregularities or extreme mispricing for deeper manual investigation.
    """)

    # 2. Global Filter
    sectors = ['All'] + sorted(df['Sector'].unique().tolist())
    col_filt, col_pad = st.columns([1, 3])
    with col_filt:
        selected_sector = st.selectbox("🌍 Filter View by Sector:", sectors)

    if selected_sector != 'All':
        summary_df = df[df['Sector'] == selected_sector].copy() # Added .copy() to avoid warnings
    else:
        summary_df = df.copy()

    # 3. High-Impact Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(summary_df)
    outliers = len(summary_df[summary_df['Status'] == 'Outlier'])
    pct = (outliers/total)*100 if total > 0 else 0
    
    col1.metric("🏢 Companies Analyzed", total)
    col2.metric("🚩 Flagged Anomalies", outliers, delta_color="inverse")
    col3.metric("⚠️ Risk Rate", f"{pct:.1f}%")
    col4.metric("📊 Model Sensitivity", "High (Z > 2.0)")
    
    st.markdown("---")

    # 4. The Market Map (Visual Hero)
    st.subheader(f"🗺️ The Forensic Market Map ({selected_sector})")
    
    with st.expander("ℹ️ How to read this map? (Click to Expand)"):
        st.markdown("""
        * **Each Dot** is a company.
        * **Position:** Determined by PCA (Principal Component Analysis). Companies with similar financial structures appear close together.
        * **Color:** <span style='color:#00F0FF'>**Cyan**</span> dots are normal. <span style='color:#FF2B2B'>**Red**</span> dots are statistical outliers.
        * **Insight:** Look for Red dots far away from the main cluster of Cyan dots. These are the "Odd Ones Out".
        """, unsafe_allow_html=True)

    if not summary_df.empty:
        # --- THE FIX IS HERE ---
        # Create a new column for size that is always positive
        summary_df['Plot_Size'] = summary_df['Anomaly_Score'].abs()
        
        fig_pca = px.scatter(
            summary_df, x='PCA_1', y='PCA_2',
            color='Status',
            color_discrete_map={'Normal': colors['normal'], 'Outlier': colors['outlier']},
            hover_name='Company Name',
            hover_data=['Sector', 'Anomaly_Score'],
            size='Plot_Size',   # <--- Updated to use the positive column
            size_max=15,
            opacity=0.8,
            title=f"Cluster Analysis: {selected_sector}"
        )
        fig_pca = apply_theme(fig_pca)
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("No data.")

    # 5. Quick Top 5 Riskiest
    st.subheader("🔥 Top 5 'Red Flags' in this View")
    if outliers > 0:
        top_risk = summary_df[summary_df['Status']=='Outlier'].sort_values(by='Anomaly_Score', ascending=False).head(5)
        st.table(top_risk[['Company Name', 'Sector', 'Anomaly_Score']])
    else:
        st.success("No anomalies detected in this sector!")

# ==========================================
# PAGE 2: THE WATCHLIST
# ==========================================
elif page == "The Watchlist":
    st.title("🚨 The Anomaly Watchlist")
    st.markdown("Companies flagged as statistical outliers. **Higher Z-Scores (Red)** indicate extreme deviation from sector norms.")
    
    outliers_df = df[df['Status'] == 'Outlier'].sort_values(by='Anomaly_Score', ascending=True)
    display_cols = ['Company Name', 'Symbol', 'Sector', 'Anomaly_Score'] + feature_cols
    
    # Custom Gradient for Dark Mode
    st.dataframe(
        outliers_df[display_cols].style.background_gradient(subset=feature_cols, cmap='RdYlGn_r'),
        use_container_width=True,
        height=600
    )

# ==========================================
# PAGE 3: SECTOR ANALYSIS
# ==========================================
elif page == "Sector Analysis":
    st.title("🏭 Sector Risk Profile")
    
    sector_stats = df.groupby('Sector').agg(
        Total=('Symbol', 'count'),
        Outliers=('Status', lambda x: (x == 'Outlier').sum())
    ).reset_index()
    sector_stats['Risk %'] = (sector_stats['Outliers'] / sector_stats['Total']) * 100
    sector_stats = sector_stats.sort_values(by='Risk %', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            sector_stats, x='Sector', y='Risk %',
            color='Risk %', color_continuous_scale='Redor',
            title="Which Sector has the most irregularities?",
            text_auto='.1f'
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.dataframe(sector_stats, hide_index=True, use_container_width=True)
    
    st.subheader("🧪 Forensic DNA by Sector")
    st.markdown("What drives outliers in each industry?")
    
    outliers_only = df[df['Status'] == 'Outlier']
    if not outliers_only.empty:
        heatmap_data = outliers_only.groupby('Sector')[feature_cols].mean()
        fig_heat = px.imshow(
            heatmap_data,
            color_continuous_scale='RdBu_r',
            title="Avg Z-Score of Outliers (Red = High Risk Factor)"
        )
        fig_heat = apply_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# PAGE 4: DEEP DIVE
# ==========================================
elif page == "Company Deep Dive":
    st.title("🔎 Forensic Deep Dive")
    
    # Combine lists so outliers appear at the top
    comps = df[df['Status']=='Outlier']['Company Name'].tolist() + df[df['Status']=='Normal']['Company Name'].tolist()
    sel_comp = st.selectbox("Select Company:", comps)
    
    row = df[df['Company Name'] == sel_comp].iloc[0]
    
    # Status Banner
    if row['Status'] == 'Outlier':
        st.error(f"⚠️ **{sel_comp}** is FLAGGED as an Anomaly (Score: {row['Anomaly_Score']:.3f})")
    else:
        st.success(f"✅ **{sel_comp}** appears NORMAL relative to peers.")

    col1, col2 = st.columns(2)
    
    # Radar Chart
    vals = [row[c] for c in feature_cols]
    vals += [vals[0]]
    thetas = [feature_names[c] for c in feature_cols]
    thetas += [thetas[0]]
    
    with col1:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=thetas, fill='toself',
            name=sel_comp,
            line_color=colors['outlier'] if row['Status']=='Outlier' else colors['normal']
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0]*len(vals), theta=thetas,
            name='Sector Avg', line=dict(color='gray', dash='dash')
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), title="Forensic Fingerprint")
        fig_radar = apply_theme(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Deviation Bar
    with col2:
        bar_df = pd.DataFrame({'Metric': [feature_names[c] for c in feature_cols], 'Z-Score': vals[:-1]})
        fig_bar = px.bar(
            bar_df, y='Metric', x='Z-Score', orientation='h',
            color='Z-Score', color_continuous_scale='RdYlGn_r',
            title="Deviation from Sector Mean (Z-Score)"
        )
        fig_bar.add_vline(x=2, line_dash="dash", line_color="white")
        fig_bar.add_vline(x=-2, line_dash="dash", line_color="white")
        fig_bar = apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 5: DATA EXPLORER
# ==========================================
elif page == "Data Explorer":
    st.title("📂 Data Explorer")

    st.dataframe(df)
