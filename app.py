import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="The Odd One Out: Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data
def load_data():
    # Load the 3 files
    try:
        universe = pd.read_csv('ind_nifty500list.csv')
        # We only really need Symbol and Company Name from the universe file
        universe = universe[['Symbol', 'Company Name']]
        
        # The model output contains almost everything we need
        watchlist = pd.read_csv('final_project_watchlist_complete.csv')
        
        # Merge to get Company Names
        data = pd.merge(watchlist, universe, on='Symbol', how='left')
        
        # Fill missing company names with Symbol if necessary
        data['Company Name'] = data['Company Name'].fillna(data['Symbol'])
        
        # Create a readable label for Anomalies
        # Logic: -1 is Anomaly, 1 is Normal
        data['Status'] = data['Anomaly_Label'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')
        
        return data
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please make sure the CSV files are in the same directory.")
        return pd.DataFrame()

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Project Navigation")
# Update this line in your existing code:
page = st.sidebar.radio("Go to", ["Executive Summary", "The Watchlist", "Sector Analysis", "Company Deep Dive", "Data Explorer"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Peer-Group Anomaly Detection\n\n"
    "**Objective:** Detecting financial statement anomalies in Indian Equities using Isolation Forests."
)

# --- Helper Lists ---
feature_cols = ['TATA_Z', 'DSRI_Z', 'AQI_Z', 'P_OCF_Z', 'PEG_Z', 'DuPont_Discrepancy_Z']
feature_names = {
    'TATA_Z': 'TATA (Accruals)',
    'DSRI_Z': 'DSRI (Receivables)',
    'AQI_Z': 'AQI (Asset Quality)',
    'P_OCF_Z': 'Price/OCF',
    'PEG_Z': 'PEG Ratio',
    'DuPont_Discrepancy_Z': 'DuPont Discrepancy'
}

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    st.title("📊 Executive Summary")
    st.markdown("### 'The Odd One Out': Detecting Financial Anomalies with AI")
    
    # --- ADDED: Sector Filter ---
    sectors = ['All'] + sorted(df['Sector'].unique().tolist())
    selected_sector = st.selectbox("Filter Dashboard by Sector:", sectors)
    
    # Filter data based on selection
    if selected_sector != 'All':
        summary_df = df[df['Sector'] == selected_sector]
    else:
        summary_df = df
    
    # Key Metrics (Calculated on filtered data)
    col1, col2, col3 = st.columns(3)
    total_companies = len(summary_df)
    total_outliers = len(summary_df[summary_df['Status'] == 'Outlier'])
    
    # Avoid division by zero if a sector has no companies
    if total_companies > 0:
        outlier_pct = (total_outliers / total_companies) * 100
    else:
        outlier_pct = 0
    
    col1.metric("Total Companies", total_companies)
    col2.metric("Anomalies Detected", total_outliers)
    col3.metric("Outlier Rate", f"{outlier_pct:.1f}%")
    
    st.markdown("---")
    
    # PCA Visualization
    st.subheader(f"Market Map: {selected_sector}")
    st.markdown(
        "This 2D map visualizes the financial similarity between companies. "
        "**Red dots** are statistically isolated companies ('Outliers')."
    )
    
    if not summary_df.empty:
        fig_pca = px.scatter(
            summary_df, 
            x='PCA_1', 
            y='PCA_2', 
            color='Status',
            # Ensure colors stay consistent even if only one group is present
            color_discrete_map={'Normal': '#1f77b4', 'Outlier': '#d62728'},
            hover_name='Company Name',
            hover_data=['Sector', 'Anomaly_Score'],
            title=f"Isolation Forest Clusters ({selected_sector})",
            template='plotly_white'
        )
        fig_pca.update_layout(height=600)
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("No data available for this sector.")

# --- PAGE 2: THE WATCHLIST ---
elif page == "The Watchlist":
    st.title("🚨 The Anomaly Watchlist")
    st.markdown("These companies have been flagged as statistical outliers based on their forensic and valuation ratios relative to their sector.")
    
    # Filter for outliers
    outliers_df = df[df['Status'] == 'Outlier'].sort_values(by='Anomaly_Score', ascending=True)
    
    # Display Options
    display_cols = ['Company Name', 'Symbol', 'Sector', 'Anomaly_Score'] + feature_cols
    
    # Formatting the dataframe for display
    st.dataframe(
        outliers_df[display_cols].style.background_gradient(subset=feature_cols, cmap='RdYlGn_r', axis=None),
        use_container_width=True,
        height=600
    )
    
    st.caption("**Note:** Ratios are Z-Scores. A value of +2.0 means the company is 2 standard deviations above its sector average. Redder cells indicate more extreme values.")

# --- PAGE: SECTOR ANALYSIS ---
elif page == "Sector Analysis":
    st.title("🏭 Sector-Level Risk Analysis")
    st.markdown("Analyze which industries display the most accounting irregularities and valuation anomalies.")
    
    # 1. League Table: Which sector has the most outliers?
    # Group by Sector and calculate stats
    sector_stats = df.groupby('Sector').agg(
        Total_Companies=('Symbol', 'count'),
        Outliers=('Status', lambda x: (x == 'Outlier').sum())
    ).reset_index()
    
    # Calculate Percentage
    sector_stats['Outlier %'] = (sector_stats['Outliers'] / sector_stats['Total_Companies']) * 100
    
    # Sort for the chart
    sector_stats = sector_stats.sort_values(by='Outlier %', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Risk Concentration by Sector")
        fig_sector_bar = px.bar(
            sector_stats, 
            x='Sector', 
            y='Outlier %',
            hover_data=['Total_Companies', 'Outliers'],
            text='Outliers', # Show count on top of bar
            color='Outlier %',
            color_continuous_scale='Reds',
            title="Percentage of Companies Flagged as Outliers"
        )
        fig_sector_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_sector_bar, use_container_width=True)
        
    with col2:
        st.subheader("Sector Summary Table")
        st.dataframe(
            sector_stats[['Sector', 'Outliers', 'Total_Companies', 'Outlier %']].style.background_gradient(subset=['Outlier %'], cmap='Reds'),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # 2. Heatmap: What drives outliers in each sector?
    st.subheader("🔥 What is driving the anomalies?")
    st.markdown(
        "This heatmap shows the **average Z-score for flagged outliers** in each sector. "
        "It reveals the dominant 'risk factor' for that industry (e.g., if the cell is Red, that ratio is typically very high for outliers in that sector)."
    )
    
    # Filter for outliers only to see what makes them special
    outliers_only = df[df['Status'] == 'Outlier']
    
    if not outliers_only.empty:
        # Calculate mean of features for outliers in each sector
        heatmap_data = outliers_only.groupby('Sector')[feature_cols].mean()
        
        # Rename columns for better readability in the chart
        heatmap_disp = heatmap_data.rename(columns=feature_names)
        
        fig_heat = px.imshow(
            heatmap_disp,
            labels=dict(x="Forensic Ratio", y="Sector", color="Avg Z-Score"),
            color_continuous_scale='RdBu_r', # Red = High, Blue = Low
            aspect="auto",
            title="Average Profile of 'Bad Apples' by Sector"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No outliers found to generate heatmap.")

# --- PAGE 3: COMPANY DEEP DIVE ---
elif page == "Company Deep Dive":
    st.title("🔎 Company Deep Dive")
    
    # Selector
    # Create a list of companies, putting outliers first for convenience
    outlier_list = df[df['Status'] == 'Outlier']['Company Name'].tolist()
    normal_list = df[df['Status'] == 'Normal']['Company Name'].tolist()
    all_companies = outlier_list + normal_list
    
    selected_company = st.selectbox("Select a Company to Investigate:", all_companies)
    
    # Get Company Data
    company_data = df[df['Company Name'] == selected_company].iloc[0]
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{company_data['Symbol']}")
        st.write(f"**Sector:** {company_data['Sector']}")
        st.write(f"**Status:** {company_data['Status']}")
        st.write(f"**Anomaly Score:** {company_data['Anomaly_Score']:.4f}")
        
        if company_data['Status'] == 'Outlier':
            st.error("⚠️ FLAGGED AS OUTLIER")
        else:
            st.success("✅ CONSIDERED NORMAL")
            
    with col2:
        # Radar Chart Logic
        values = [company_data[col] for col in feature_cols]
        # Close the loop for radar chart
        r_values = values + [values[0]]
        theta_values = [feature_names[col] for col in feature_cols]
        theta = theta_values + [theta_values[0]]
        
        fig_radar = go.Figure()
        
        # Add the company trace
        fig_radar.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta,
            fill='toself',
            name=selected_company,
            line_color='red' if company_data['Status'] == 'Outlier' else 'blue'
        ))
        
        # Add a "Normal Range" circle (e.g., Z=0)
        fig_radar.add_trace(go.Scatterpolar(
            r=[0] * len(r_values),
            theta=theta,
            name='Sector Average',
            line_color='green',
            line_dash='dash'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-3, 3] # Set fixed range for Z-scores for better comparison
                )
            ),
            showlegend=True,
            title=f"Forensic Profile vs Sector Average (Z-Scores)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Bar Chart for Contribution
    st.subheader("Feature Deviation Analysis")
    
    # Prepare data for bar chart
    bar_data = pd.DataFrame({
        'Metric': [feature_names[col] for col in feature_cols],
        'Z-Score': values
    })
    
    fig_bar = px.bar(
        bar_data, 
        x='Z-Score', 
        y='Metric', 
        orientation='h',
        color='Z-Score',
        color_continuous_scale='RdYlGn_r',
        title="Which Ratios are Driving the Anomaly?"
    )
    # Add vertical lines for +/- 2 SD
    fig_bar.add_vline(x=2, line_dash="dash", line_color="gray", annotation_text="+2 SD")
    fig_bar.add_vline(x=-2, line_dash="dash", line_color="gray", annotation_text="-2 SD")
    
    st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE 4: DATA EXPLORER ---
elif page == "Data Explorer":
    st.title("📂 Data Explorer")
    
    # Filter by Sector
    sectors = ['All'] + sorted(df['Sector'].unique().tolist())
    selected_sector = st.selectbox("Filter by Sector:", sectors)
    
    if selected_sector != 'All':
        display_df = df[df['Sector'] == selected_sector]
    else:
        display_df = df
        
    st.dataframe(display_df)
    
    # Download Button
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(display_df)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='anomaly_detection_data.csv',
        mime='text/csv',
    )