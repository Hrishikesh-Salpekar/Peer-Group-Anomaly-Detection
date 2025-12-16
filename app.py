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
    st.info("""
    **👋 What is this dashboard?** This tool uses **Unsupervised Machine Learning (Isolation Forests)** to scan the Nifty 500. 
    It identifies companies whose financial accounting ratios (Accruals, Asset Quality, Valuation) 
    **deviate significantly** from their sector peers. 
    
    **🎯 Goal:** Flag potential accounting irregularities or extreme mispricing for deeper manual investigation.
    """)

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
# PAGE 3: SECTOR ANALYSIS (ENHANCED)
# ==========================================
elif page == "Sector Analysis":
    st.title("🏭 Sector Risk Profile")
    st.markdown("Deep dive into *where* the risks are concentrating and *what* is driving them.")
    
    # --- 1. SECTOR LEAGUE TABLE (Existing) ---
    sector_stats = df.groupby('Sector').agg(
        Total=('Symbol', 'count'), 
        Outliers=('Status', lambda x: (x == 'Outlier').sum())
    ).reset_index()
    sector_stats['Risk %'] = (sector_stats['Outliers'] / sector_stats['Total']) * 100
    sector_stats = sector_stats.sort_values(by='Risk %', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_risk = px.bar(
            sector_stats, x='Sector', y='Risk %', 
            color='Risk %', color_continuous_scale='Redor', 
            title="Percentage of Companies Flagged per Sector",
            text_auto='.1f'
        )
        fig_risk = apply_theme(fig_risk)
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with col2:
        st.caption("Sector Summary Data")
        st.dataframe(
            sector_stats[['Sector', 'Outliers', 'Total', 'Risk %']].style.background_gradient(subset=['Risk %'], cmap='Reds'),
            hide_index=True, 
            use_container_width=True,
            height=350
        )

    st.markdown("---")

    # --- 2. "THE WHY" CHART (New!) ---
    st.subheader("🧩 The 'Why' Analysis: Drivers of Anomalies")
    st.markdown("We know *which* sectors have outliers. This chart explains **why** they were flagged.")
    
    # Logic to identify primary driver for every outlier
    outliers_only = df[df['Status'] == 'Outlier'].copy()
    
    if not outliers_only.empty:
        # Function to get the column name of the max Z-score
        def get_driver(row):
            return feature_names[row[feature_cols].abs().idxmax()]
        
        outliers_only['Primary Driver'] = outliers_only.apply(get_driver, axis=1)
        
        # Group by Sector and Driver
        driver_stats = outliers_only.groupby(['Sector', 'Primary Driver']).size().reset_index(name='Count')
        
        fig_stacked = px.bar(
            driver_stats, 
            x='Sector', y='Count', color='Primary Driver',
            title="Breakdown of Anomaly Causes by Sector",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            barmode='stack'
        )
        fig_stacked = apply_theme(fig_stacked)
        st.plotly_chart(fig_stacked, use_container_width=True)
    else:
        st.info("No outliers found to analyze drivers.")

    # --- 3. "THE SPREAD" CHART (New!) ---
    st.subheader("📊 Systemic Risk vs. One-Off Events")
    st.markdown("""
    * **Tight Box:** The sector is homogeneous (everyone behaves similarly).
    * **Tall Box / Long Whiskers:** The sector is volatile with varied financial practices.
    * **Dots far above:** Extreme outliers in an otherwise normal sector.
    """)
    
    # Box plot of Anomaly Scores by Sector
    fig_box = px.box(
        df, x='Sector', y='Anomaly_Score', 
        color='Sector', 
        points='outliers', # Only show points that are outliers
        title="Distribution of Anomaly Scores (Volatility check)"
    )
    fig_box.update_layout(showlegend=False)
    fig_box = apply_theme(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# PAGE 4: DEEP DIVE (SMART ANALYTICS)
# ==========================================
elif page == "Company Deep Dive":
    st.title("🔎 Forensic Deep Dive")
    
    # 1. Company Selector (Outliers First)
    comps = df[df['Status']=='Outlier']['Company Name'].tolist() + df[df['Status']=='Normal']['Company Name'].tolist()
    sel_comp = st.selectbox("Select Company:", comps)
    
    # Get Data
    row = df[df['Company Name'] == sel_comp].iloc[0]
    symbol = row['Symbol']
    symbol = str(symbol+".NS")    

    # --- 2. KPI HEADER (Clean & Professional) ---
    # Calculate Primary Risk Driver
    vals = row[feature_cols]
    max_col = vals.abs().idxmax() # e.g., 'TATA_Z'
    max_val = vals[max_col]       # e.g., 2.5
    driver_name = feature_names[max_col]
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Sector", row['Sector'])
    kpi2.metric("Anomaly Score", f"{row['Anomaly_Score']:.3f}", delta="Lower is Better" if row['Status']=='Normal' else "High Risk", delta_color="inverse")
    kpi3.metric("Primary Risk Driver", driver_name)
    kpi4.metric("Deviation Level", f"{max_val:.1f} σ", help="Standard Deviations from Sector Mean")

    st.markdown("---")
    
    # --- 3. DYNAMIC INSIGHT & HISTORICAL EVIDENCE ---
    col_insight, col_chart = st.columns([1, 2])
    
    # Prepare Historical Data for this company
    comp_hist = pd.DataFrame()
    if not hist_df.empty:
        comp_hist = hist_df[hist_df['Symbol'] == symbol].sort_values(by='Year')

    with col_insight:
        st.subheader("📝 Forensic Insight")
        
        # Define logic for the "Insight" text and "Evidence" metrics based on the Risk Driver
        if max_col == 'TATA_Z': # Accruals
            st.warning(f"**High Accruals Detected.** {sel_comp} is reporting profits that are not backed by operating cash flow.")
            if not comp_hist.empty:
                latest = comp_hist.iloc[-1]
                st.metric("Latest Net Income", f"₹{latest['Net Income']/1e7:.1f} Cr")
                st.metric("Latest OCF", f"₹{latest['OCF']/1e7:.1f} Cr", delta=f"Gap: {(latest['OCF']-latest['Net Income'])/1e7:.1f} Cr", delta_color="off")
        
        elif max_col == 'DSRI_Z': # Receivables
            st.warning(f"**Aggressive Revenue Recognition?** Receivables are growing significantly faster than Sales.")
            if not comp_hist.empty:
                latest = comp_hist.iloc[-1]
                prev = comp_hist.iloc[-2] if len(comp_hist) > 1 else latest
                
                rev_growth = ((latest['Revenue'] - prev['Revenue']) / prev['Revenue']) * 100
                rec_growth = ((latest['Receivables'] - prev['Receivables']) / prev['Receivables']) * 100 if 'Receivables' in latest else 0
                
                st.metric("Revenue Growth (YoY)", f"{rev_growth:.1f}%")
                st.metric("Receivables Growth (YoY)", f"{rec_growth:.1f}%", delta=f"Excess: {rec_growth-rev_growth:.1f}%", delta_color="inverse")

        elif max_col == 'DuPont_Discrepancy_Z': # ROE / Growth
            st.warning(f"**DuPont Mismatch.** The ROE being reported may be unsustainable given the equity base.")
            if not comp_hist.empty and 'Total Equity' in comp_hist.columns:
                latest = comp_hist.iloc[-1]
                roe = (latest['Net Income'] / latest['Total Equity']) * 100
                st.metric("Return on Equity (ROE)", f"{roe:.1f}%")
                st.metric("Equity Base", f"₹{latest['Total Equity']/1e7:.1f} Cr")

        else: # Default/Other
            st.info(f"The primary deviation is in **{driver_name}**. This warrants a review of the company's valuation or asset structure relative to peers.")

    with col_chart:
        st.subheader("📉 Supporting Evidence (Trend)")
        
        if not comp_hist.empty:
            # DYNAMIC CHART SELECTION
            if max_col == 'TATA_Z':
                # Plot Income vs Cash Flow (The classic Accrual check)
                fig = px.bar(comp_hist, x='Year', y=['Net Income', 'OCF'], barmode='group',
                             title="Earnings Quality: Profit (Blue) vs Cash (Red)",
                             color_discrete_map={'Net Income': '#00F0FF', 'OCF': '#FF2B2B'})
            
            elif max_col == 'DSRI_Z' and 'Receivables' in comp_hist.columns:
                # Plot Revenue vs Receivables (Normalized or dual axis ideally, but line chart works)
                fig = px.line(comp_hist, x='Year', y=['Revenue', 'Receivables'], markers=True,
                              title="Revenue vs Receivables Growth",
                              color_discrete_sequence=['#00F0FF', '#FFD700'])
            
            elif max_col == 'DuPont_Discrepancy_Z' and 'Total Equity' in comp_hist.columns:
                # Plot ROE Components
                fig = px.line(comp_hist, x='Year', y=['Net Income', 'Total Equity'], markers=True,
                              title="ROE Components: Income vs Equity Base",
                              color_discrete_sequence=['#00F0FF', '#00FF00'])
                
            else:
                # Default Chart (Revenue vs Profit)
                fig = px.line(comp_hist, x='Year', y=['Revenue', 'Net Income'], markers=True,
                              title="General Financial Performance",
                              color_discrete_sequence=['#00F0FF', '#FFD700'])
            
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available to plot the specific anomaly trend.")

    # --- 4. THE RADAR SCAN (Kept for completeness) ---
    st.markdown("### 🕸️ Full Forensic Fingerprint")
    vals = [row[c] for c in feature_cols]
    vals += [vals[0]]
    thetas = [feature_names[c] for c in feature_cols]
    thetas += [thetas[0]]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=thetas, fill='toself', name=sel_comp,
        line_color=colors['outlier'] if row['Status']=='Outlier' else colors['normal']
    ))
    fig_radar.add_trace(go.Scatterpolar(r=[0]*len(vals), theta=thetas, name='Sector Avg', line=dict(color='gray', dash='dash')))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), title=f"{sel_comp} vs Sector Norms")
    fig_radar = apply_theme(fig_radar)
    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# PAGE 5: DATA EXPLORER
# ==========================================
elif page == "Data Explorer":
    st.title("📂 Data Explorer")
    st.dataframe(df)