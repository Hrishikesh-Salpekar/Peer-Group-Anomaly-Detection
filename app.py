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

# --- 🎨 Theme Definition ---
colors = {
    'background': '#0E1117',
    'text': '#FAFAFA',
    'normal': '#00F0FF',   # Neon Cyan
    'outlier': '#FF2B2B',  # Neon Red
    'accent': '#FFD700',   # Gold
    'card_bg': '#1E1E1E'
}

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
        data['Status'] = data['Anomaly_Label'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')
        return data
    except FileNotFoundError:
        st.error("❌ Data files missing. Please upload the CSVs.")
        return pd.DataFrame()

@st.cache_data
def load_historical_data():
    try:
        # Load the user's new file format
        hist_df = pd.read_csv('nifty500_financials.csv')
        
        # --- PRE-PROCESSING FOR DASHBOARD ---
        # 1. Convert Date to Year
        if 'Financial Year End' in hist_df.columns:
            hist_df['Year'] = pd.to_datetime(hist_df['Financial Year End']).dt.year
        
        # 2. Rename columns to standard names for easier plotting
        # Mapping: New File Header -> Dashboard Internal Name
        column_map = {
            'Ticker': 'Symbol',
            'Total Revenue': 'Revenue',
            'Operating Cash Flow (CFO)': 'OCF'
        }
        hist_df = hist_df.rename(columns=column_map)
        
        return hist_df
    except FileNotFoundError:
        return pd.DataFrame() # Return empty if file not found

df = load_data()
hist_df = load_historical_data()

# --- Sidebar ---
st.sidebar.title("🕵️‍♂️ Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "The Watchlist", "Sector Analysis", "Company Deep Dive", "Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **Model:** Isolation Forest")
st.sidebar.caption("📊 **Data:** Nifty 500 (TTM)")

feature_cols = ['TATA_Z', 'DSRI_Z', 'AQI_Z', 'P_OCF_Z', 'PEG_Z', 'DuPont_Discrepancy_Z']
feature_names = {
    'TATA_Z': 'Accruals (TATA)', 'DSRI_Z': 'Receivables (DSRI)',
    'AQI_Z': 'Asset Quality (AQI)', 'P_OCF_Z': 'Price/CashFlow',
    'PEG_Z': 'PEG Ratio', 'DuPont_Discrepancy_Z': 'DuPont Discrepancy'
}

# --- Risk Narrative Logic ---
risk_map = {
    'TATA_Z': {
        'high': "High Accruals: Profits may not be backed by cash flow (Earnings Quality Risk).",
        'low': "Low Accruals: Cash flow is strong relative to reported profit."
    },
    'DSRI_Z': {
        'high': "High Receivables Growth: Revenue might be driven by aggressive credit sales.",
        'low': "Low Receivables: Efficient collection cycle."
    },
    'AQI_Z': {
        'high': "High Asset Quality Index: Potential capitalization of expenses into non-current assets.",
        'low': "Stable Asset Base."
    },
    'P_OCF_Z': {
        'high': "High Price-to-CashFlow: Valuation appears disconnected from cash generation.",
        'low': "Low Valuation relative to Cash Flow."
    },
    'PEG_Z': {
        'high': "High PEG Ratio: Expensive valuation relative to growth rate.",
        'low': "Undervalued relative to growth."
    },
    'DuPont_Discrepancy_Z': {
        'high': "DuPont Discrepancy: Significant gap between ROE and Sustainable Growth Rate.",
        'low': "Consistent Growth Dynamics."
    }
}

def get_risk_insight(row):
    # Get the feature columns
    vals = row[feature_cols]
    # Find the feature with the Maximum Absolute Z-Score
    max_col = vals.abs().idxmax()
    max_val = vals[max_col]
    
    # Determine direction
    direction = 'high' if max_val > 0 else 'low'
    
    # Get the narrative
    narrative = risk_map.get(max_col, {}).get(direction, "Analysis inconclusive.")
    
    return max_col, max_val, narrative

# ==========================================
# PAGE 1: EXECUTIVE SUMMARY
# ==========================================
if page == "Executive Summary":
    st.title("🛡️ Peer-Group Anomaly Detection")
    st.markdown("<h3 style='color:#00F0FF; margin-top:-20px'>AI-Driven Forensic Audit of Indian Equities</h3>", unsafe_allow_html=True)
    st.info("**🎯 Goal:** Identify companies whose financial ratios deviate significantly from their sector peers.")

    sectors = ['All'] + sorted(df['Sector'].unique().tolist())
    selected_sector = st.selectbox("🌍 Filter View by Sector:", sectors)

    summary_df = df[df['Sector'] == selected_sector].copy() if selected_sector != 'All' else df.copy()

    col1, col2, col3, col4 = st.columns(4)
    total = len(summary_df)
    outliers = len(summary_df[summary_df['Status'] == 'Outlier'])
    pct = (outliers/total)*100 if total > 0 else 0
    
    col1.metric("Companies", total)
    col2.metric("Anomalies", outliers, delta_color="inverse")
    col3.metric("Risk Rate", f"{pct:.1f}%")
    col4.metric("Model Sensitivity", "Z > 2.0")
    
    st.markdown("---")
    st.subheader(f"🗺️ The Forensic Market Map ({selected_sector})")
    
    if not summary_df.empty:
        summary_df['Plot_Size'] = summary_df['Anomaly_Score'].abs()
        fig_pca = px.scatter(
            summary_df, x='PCA_1', y='PCA_2', color='Status',
            color_discrete_map={'Normal': colors['normal'], 'Outlier': colors['outlier']},
            hover_name='Company Name', hover_data=['Sector', 'Anomaly_Score'],
            size='Plot_Size', size_max=15, opacity=0.8,
            title=f"Cluster Analysis: {selected_sector}"
        )
        fig_pca = apply_theme(fig_pca)
        st.plotly_chart(fig_pca, use_container_width=True)

    if outliers > 0:
        st.subheader("🔥 Top 5 Highest Risk Companies")
        st.table(summary_df[summary_df['Status']=='Outlier'].sort_values(by='Anomaly_Score', ascending=False).head(5)[['Company Name', 'Sector', 'Anomaly_Score']])

# ==========================================
# PAGE 2: THE WATCHLIST
# ==========================================
elif page == "The Watchlist":
    st.title("🚨 The Anomaly Watchlist")
    outliers_df = df[df['Status'] == 'Outlier'].sort_values(by='Anomaly_Score', ascending=True)
    display_cols = ['Company Name', 'Symbol', 'Sector', 'Anomaly_Score'] + feature_cols
    st.dataframe(outliers_df[display_cols].style.background_gradient(subset=feature_cols, cmap='RdYlGn_r'), use_container_width=True, height=600)

# ==========================================
# PAGE 3: SECTOR ANALYSIS
# ==========================================
elif page == "Sector Analysis":
    st.title("🏭 Sector Risk Profile")
    sector_stats = df.groupby('Sector').agg(
        Total=('Symbol', 'count'), Outliers=('Status', lambda x: (x == 'Outlier').sum())
    ).reset_index()
    sector_stats['Risk %'] = (sector_stats['Outliers'] / sector_stats['Total']) * 100
    sector_stats = sector_stats.sort_values(by='Risk %', ascending=False)
    
    fig = px.bar(sector_stats, x='Sector', y='Risk %', color='Risk %', color_continuous_scale='Redor', title="Sector Risk Concentration")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 4: DEEP DIVE (UPDATED)
# ==========================================
elif page == "Company Deep Dive":
    st.title("🔎 Forensic Deep Dive")
    
    # Sort outliers to top
    comps = df[df['Status']=='Outlier']['Company Name'].tolist() + df[df['Status']=='Normal']['Company Name'].tolist()
    sel_comp = st.selectbox("Select Company:", comps)
    
    row = df[df['Company Name'] == sel_comp].iloc[0]
    
    # --- 1. SMART INSIGHT BANNER ---
    st.subheader("Diagnostic Report")
    max_col, max_val, narrative = get_risk_insight(row)
    
    if row['Status'] == 'Outlier':
        st.error(f"⚠️ **FLAGGED AS ANOMALY** (Score: {row['Anomaly_Score']:.3f})")
        st.markdown(f"""
        <div style="background-color: #262730; padding: 15px; border-radius: 5px; border-left: 5px solid #FF2B2B;">
            <h4 style="margin:0">Primary Risk Driver: {feature_names[max_col]}</h4>
            <p style="font-size: 16px; margin-top: 5px;">
            <b>Observation:</b> Z-Score is <b>{max_val:.2f}</b> (Standard Deviations from Mean).<br>
            <b>Insight:</b> {narrative}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ **NORMAL** (Score: {row['Anomaly_Score']:.3f})")
        st.info(f"The company's financial profile is consistent with sector peers. The most deviant metric is {feature_names[max_col]} (Z={max_val:.2f}), which is within acceptable limits.")

    st.markdown("---")
    
    # --- 2. FORENSIC CHARTS ---
    col1, col2 = st.columns(2)
    
    vals = [row[c] for c in feature_cols]
    vals += [vals[0]]
    thetas = [feature_names[c] for c in feature_cols]
    thetas += [thetas[0]]
    
    with col1:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=thetas, fill='toself', name=sel_comp,
            line_color=colors['outlier'] if row['Status']=='Outlier' else colors['normal']
        ))
        fig_radar.add_trace(go.Scatterpolar(r=[0]*len(vals), theta=thetas, name='Sector Avg', line=dict(color='gray', dash='dash')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), title="Forensic Fingerprint")
        fig_radar = apply_theme(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        bar_df = pd.DataFrame({'Metric': [feature_names[c] for c in feature_cols], 'Z-Score': vals[:-1]})
        fig_bar = px.bar(
            bar_df, y='Metric', x='Z-Score', orientation='h',
            color='Z-Score', color_continuous_scale='RdYlGn_r',
            title="Deviation from Sector Mean"
        )
        fig_bar.add_vline(x=2, line_dash="dash", line_color="white")
        fig_bar.add_vline(x=-2, line_dash="dash", line_color="white")
        fig_bar = apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 3. HISTORICAL DATA (DYNAMIC) ---
    st.subheader("📈 Historical Financial Trend")
    
    if not hist_df.empty:
        # Match Symbol from Main Data to Ticker in Historical Data
        # Ensure we check for both "RELIANCE" and "RELIANCE.NS" style matches if needed
        # For now, we assume strict matching based on your universe file
        symbol = row['Symbol']
        
        # Try finding the symbol in the historical data
        comp_hist = hist_df[hist_df['Symbol'] == symbol].sort_values(by='Year')
        
        if not comp_hist.empty:
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                # Plot Revenue vs Net Income
                # We need to make sure the columns exist and are numeric
                if 'Revenue' in comp_hist.columns and 'Net Income' in comp_hist.columns:
                     fig_trend = px.line(comp_hist, x='Year', y=['Revenue', 'Net Income'], markers=True, 
                                        title="Revenue vs Net Income (Historical)", 
                                        color_discrete_sequence=['#00F0FF', '#FFD700'])
                     fig_trend = apply_theme(fig_trend)
                     st.plotly_chart(fig_trend, use_container_width=True)
            
            with col_h2:
                # Plot OCF Trend
                if 'OCF' in comp_hist.columns:
                    fig_ocf = px.bar(comp_hist, x='Year', y='OCF', 
                                    title="Operating Cash Flow Trend", 
                                    color_discrete_sequence=['#00FF00'])
                    fig_ocf = apply_theme(fig_ocf)
                    st.plotly_chart(fig_ocf, use_container_width=True)
        else:
            st.warning(f"No historical data found for {symbol} in the uploaded file.")
            st.caption(f"Available tickers in file: {', '.join(hist_df['Symbol'].unique()[:5])}...")
    else:
        st.info("💡 **Want to see charts?** Upload a file named `nifty500_financials_test.csv` (or similar) with columns like 'Ticker', 'Total Revenue', etc.")

# ==========================================
# PAGE 5: DATA EXPLORER
# ==========================================
elif page == "Data Explorer":
    st.title("📂 Data Explorer")
    st.dataframe(df)
