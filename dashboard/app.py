"""
Marketing Intelligence Suite — Interactive Streamlit Dashboard
Run with: streamlit run dashboard/app.py

Dataset: Online Retail II UCI (place at data/raw/online_retail_II.xlsx)
Falls back to synthetic data if processed CSVs from notebooks are not yet available.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing Intelligence Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1f2328; }
    .metric-label { font-size: 0.85rem; color: #57606a; margin-top: 4px; }
    .section-title { color: #1f2328; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/48/000000/analytics.png", width=48)
st.sidebar.title("Marketing Intelligence Suite")
st.sidebar.markdown("*Arshia Fajar · Data Science Portfolio*")
st.sidebar.divider()

page = st.sidebar.selectbox(
    "Navigate to Module",
    ["Overview", "Customer Segmentation", "Propensity & CLV", "A/B Testing", "NLP & Sentiment"]
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset Parameters**")
n_customers = st.sidebar.slider("Synthetic customers", 500, 5000, 2000, step=500)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# ── Data Generation (cached) ──────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

@st.cache_data
def generate_data(n, seed):
    # ── Prefer real data from Module 1 + 2 outputs ───────────────────────
    rfm_path  = os.path.join(PROCESSED_DIR, 'rfm_segments.csv')
    pred_path = os.path.join(PROCESSED_DIR, 'propensity_clv_predictions.csv')

    if os.path.exists(rfm_path):
        rfm = pd.read_csv(rfm_path)
        if 'AvgOrderValue' not in rfm.columns:
            rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
        if 'FrequencyRecency' not in rfm.columns:
            rfm['FrequencyRecency'] = rfm['Frequency'] / (rfm['Recency'] + 1)
        # Merge predictions if available
        if os.path.exists(pred_path):
            preds = pd.read_csv(pred_path)
            rfm = rfm.merge(preds[['CustomerID','PropensityScore','PredictedCLV','WillReturn_Pred']],
                             on='CustomerID', how='left')
            rfm.rename(columns={'WillReturn_Pred': 'WillReturn'}, inplace=True)
        if 'PropensityScore' not in rfm.columns:
            log_odds = -0.01*rfm['Recency'] + 0.15*rfm['Frequency'] + 0.002*rfm['Monetary'] - 2
            rfm['PropensityScore'] = 1 / (1 + np.exp(-log_odds))
        if 'WillReturn' not in rfm.columns:
            rfm['WillReturn'] = (rfm['Recency'] <= 90).astype(int)
        if 'PredictedCLV' not in rfm.columns:
            rfm['PredictedCLV'] = (rfm['AvgOrderValue'] * rfm['FrequencyRecency'] * 90).clip(0)
        if 'Segment' not in rfm.columns:
            rfm['Segment'] = 'Uncategorised'
        return rfm

    # ── Synthetic fallback ────────────────────────────────────────────────
    np.random.seed(seed)
    rfm = pd.DataFrame({
        'CustomerID': range(10000, 10000 + n),
        'Recency':    np.random.exponential(60, n).astype(int).clip(1, 365),
        'Frequency':  np.random.negative_binomial(3, 0.4, n).clip(1, 50),
        'Monetary':   np.round(np.random.lognormal(5, 1.2, n), 2),
    })
    rfm['AvgOrderValue']    = rfm['Monetary'] / rfm['Frequency']
    rfm['FrequencyRecency'] = rfm['Frequency'] / (rfm['Recency'] + 1)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(np.log1p(rfm[['Recency', 'Frequency', 'Monetary']]))
    km = KMeans(n_clusters=4, random_state=seed, n_init=10)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().rank().astype(int) - 1
    rfm['Segment'] = rfm['Cluster'].map(cluster_means).map(
        {0: 'At-Risk', 1: 'New Customers', 2: 'Potential Loyalists', 3: 'Champions'}
    )
    log_odds = -0.01*rfm['Recency'] + 0.15*rfm['Frequency'] + 0.002*rfm['Monetary'] - 2
    rfm['PropensityScore'] = 1 / (1 + np.exp(-log_odds))
    rfm['WillReturn']   = (np.random.random(n) < rfm['PropensityScore']).astype(int)
    rfm['PredictedCLV'] = (rfm['AvgOrderValue'] * rfm['Frequency'] * 0.4
                           + rfm['FrequencyRecency'] * 50).clip(0)
    return rfm

@st.cache_data
def generate_ab_data(n, seed):
    np.random.seed(seed)
    n_grp = n
    ctrl = pd.DataFrame({
        'Group': 'Control', 'Converted': np.random.binomial(1, 0.12, n_grp),
        'OrderValue': np.round(np.random.lognormal(3.8, 0.7, n_grp), 2)
    })
    trt = pd.DataFrame({
        'Group': 'Treatment', 'Converted': np.random.binomial(1, 0.155, n_grp),
        'OrderValue': np.round(np.random.lognormal(3.95, 0.7, n_grp), 2)
    })
    ctrl.loc[ctrl['Converted'] == 0, 'OrderValue'] = 0
    trt.loc[trt['Converted'] == 0, 'OrderValue'] = 0
    return pd.concat([ctrl, trt], ignore_index=True)

@st.cache_data
def generate_reviews(n, seed):
    np.random.seed(seed)
    templates = [
        "Fast delivery and the product arrived in perfect condition.",
        "Shipping was incredibly quick. Will definitely order again!",
        "Fantastic quality product, well worth the money.",
        "Customer service was amazing. Great experience overall.",
        "Product is okay, nothing special but does the job.",
        "Shipping took nearly two weeks! Way too slow.",
        "Poor quality product, broke after two days. Avoid!",
        "Customer service was unhelpful and dismissive.",
        "Great value for money! Highly recommended.",
        "Nothing like the description on the website.",
    ]
    sentiments = ['Positive','Positive','Positive','Positive','Neutral','Negative','Negative','Negative','Positive','Negative']
    topics = ['Delivery','Product Quality','Product Quality','Customer Service','Mixed',
              'Delivery','Product Quality','Customer Service','Value','Product Accuracy']
    scores = [0.72, 0.85, 0.78, 0.82, 0.05, -0.65, -0.80, -0.55, 0.90, -0.60]

    idx = np.random.randint(0, len(templates), n)
    noise = np.random.normal(0, 0.05, n)
    return pd.DataFrame({
        'ReviewText':      [templates[i] for i in idx],
        'SentimentLabel':  [sentiments[i] for i in idx],
        'Topic':           [topics[i] for i in idx],
        'CompoundScore':   np.clip([scores[i] + noise[j] for j, i in enumerate(idx)], -1, 1),
        'StarRating':      np.random.choice([1,2,3,4,5], n, p=[0.08,0.07,0.15,0.35,0.35])
    })

rfm = generate_data(n_customers, random_seed)
ab  = generate_ab_data(n_customers // 2, random_seed)
rev = generate_reviews(min(n_customers, 2000), random_seed)

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Marketing Intelligence Suite")
    st.markdown("*End-to-end data science solution for marketing analytics — Arshia Fajar*")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Analysed", f"{n_customers:,}")
    col2.metric("Avg Customer LTV",   f"£{rfm['PredictedCLV'].mean():.0f}")
    col3.metric("Return Propensity",  f"{rfm['WillReturn'].mean()*100:.1f}%")
    col4.metric("Positive Reviews",   f"{(rev['SentimentLabel']=='Positive').mean()*100:.0f}%")

    st.divider()
    st.subheader("Project Modules")
    cols = st.columns(4)
    modules = [
        ("Customer Segmentation", "RFM · K-Means · DBSCAN · PCA · Silhouette Score"),
        ("Propensity & CLV",      "Gradient Boosting · Cross-Validation · GridSearchCV"),
        ("A/B Testing",           "Chi-Square · t-test · Cohen's d · Causal Inference"),
        ("NLP & Sentiment",       "VADER · LDA Topics · GenAI Summarization"),
    ]
    for col, (title, tech) in zip(cols, modules):
        with col:
            st.markdown(f"### {title}")
            st.caption(tech)

# ── Page: Customer Segmentation ───────────────────────────────────────────────
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.markdown("RFM-based segmentation using K-Means clustering")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Segment Distribution")
        seg_counts = rfm['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#3b82d4', '#27ae60', '#f39c12', '#e74c3c']
        bars = ax.bar(seg_counts.index, seg_counts.values, color=colors[:len(seg_counts)], edgecolor='white')
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(val), ha='center', fontweight='bold')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customers per Segment', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Segment Profiles")
        profile = rfm.groupby('Segment').agg(
            Customers=('CustomerID','count'),
            Avg_Recency=('Recency','mean'),
            Avg_Frequency=('Frequency','mean'),
            Avg_Monetary=('Monetary','mean')
        ).round(1)
        st.dataframe(profile, use_container_width=True)

    st.subheader("RFM Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col_name, color in zip(axes, ['Recency','Frequency','Monetary'],
                                    ['#3b82d4','#27ae60','#f39c12']):
        rfm[col_name].hist(bins=40, ax=ax, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(f'{col_name} Distribution', fontweight='bold')
        ax.axvline(rfm[col_name].mean(), color='red', linestyle='--', label='Mean')
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Page: Propensity & CLV ─────────────────────────────────────────────────────
elif page == "Propensity & CLV":
    st.title("Customer Propensity & CLV Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("High-Propensity Customers", f"{(rfm['PropensityScore']>0.7).sum():,}")
    col2.metric("Avg Predicted CLV", f"£{rfm['PredictedCLV'].mean():.2f}")
    col3.metric("Top 20% CLV",       f"£{rfm['PredictedCLV'].quantile(0.8):.2f}+")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Propensity Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rfm['PropensityScore'], bins=40, color='#3b82d4', edgecolor='white', alpha=0.8)
        ax.axvline(0.5, color='red', linestyle='--', label='Decision threshold')
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("CLV by Customer Segment")
        fig, ax = plt.subplots(figsize=(7, 4))
        rfm.boxplot(column='PredictedCLV', by='Segment', ax=ax)
        plt.suptitle('')
        ax.set_title('Predicted CLV by Segment', fontweight='bold')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Predicted CLV (£)')
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close()

    st.subheader("Top 20 Highest-Value Customers")
    top20 = rfm.nlargest(20, 'PredictedCLV')[
        ['CustomerID','Segment','Recency','Frequency','Monetary','PropensityScore','PredictedCLV']
    ].round(2)
    st.dataframe(top20, use_container_width=True)

# ── Page: A/B Testing ─────────────────────────────────────────────────────────
elif page == "A/B Testing":
    from scipy.stats import chi2_contingency, ttest_ind
    st.title("A/B Test Analysis — Email Campaign")

    ctrl = ab[ab['Group']=='Control']
    trt  = ab[ab['Group']=='Treatment']

    cr_ctrl = ctrl['Converted'].mean()
    cr_trt  = trt['Converted'].mean()
    lift    = (cr_trt - cr_ctrl) / cr_ctrl * 100

    contingency = np.array([
        [ctrl['Converted'].sum(), len(ctrl) - ctrl['Converted'].sum()],
        [trt['Converted'].sum(),  len(trt)  - trt['Converted'].sum()]
    ])
    _, p_chi2, _, _ = chi2_contingency(contingency)

    ctrl_buyers = ctrl[ctrl['Converted']==1]['OrderValue']
    trt_buyers  = trt[trt['Converted']==1]['OrderValue']
    _, p_ttest  = ttest_ind(ctrl_buyers, trt_buyers, equal_var=False)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Conv. Rate",   f"{cr_ctrl*100:.2f}%")
    col2.metric("Treatment Conv. Rate", f"{cr_trt*100:.2f}%", delta=f"+{lift:.1f}% lift")
    col3.metric("Conversion p-value",   f"{p_chi2:.4f}", delta="Significant" if p_chi2 < 0.05 else "n.s.")
    col4.metric("AOV p-value",          f"{p_ttest:.4f}", delta="Significant" if p_ttest < 0.05 else "n.s.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conversion Rate Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Control', 'Treatment'], [cr_ctrl*100, cr_trt*100],
                      color=['#3b82d4','#e74c3c'], edgecolor='white', width=0.5)
        for bar, val in zip(bars, [cr_ctrl, cr_trt]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f'{val*100:.2f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('A/B Test: Conversion Rate', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Order Value Distribution (Converters)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ctrl_buyers, bins=30, alpha=0.6, label='Control',   color='#3b82d4', density=True)
        ax.hist(trt_buyers,  bins=30, alpha=0.6, label='Treatment', color='#e74c3c', density=True)
        ax.axvline(ctrl_buyers.mean(), color='#3b82d4', linestyle='--', lw=2)
        ax.axvline(trt_buyers.mean(),  color='#e74c3c', linestyle='--', lw=2)
        ax.set_xlabel('Order Value (£)')
        ax.legend()
        ax.set_title('AOV Distribution', fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ── Page: NLP & Sentiment ─────────────────────────────────────────────────────
elif page == "NLP & Sentiment":
    st.title("NLP: Sentiment Analysis & Topic Modeling")

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviews Analyzed",  f"{len(rev):,}")
    col2.metric("Positive Reviews",  f"{(rev['SentimentLabel']=='Positive').sum():,}")
    col3.metric("Negative Reviews",  f"{(rev['SentimentLabel']=='Negative').sum():,}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sent_counts = rev['SentimentLabel'].value_counts()
        colors_map = {'Positive':'#27ae60','Neutral':'#f39c12','Negative':'#e74c3c'}
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(sent_counts.index, sent_counts.values,
                      color=[colors_map.get(s,'#888') for s in sent_counts.index],
                      edgecolor='white')
        for bar, val in zip(bars, sent_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(val), ha='center', fontweight='bold')
        ax.set_ylabel('Reviews')
        ax.set_title('Sentiment Breakdown', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Avg Sentiment by Topic")
        topic_sent = rev.groupby('Topic')['CompoundScore'].mean().sort_values()
        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in topic_sent.values]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(topic_sent.index, topic_sent.values, color=colors, edgecolor='white')
        ax.axvline(0, color='black', linestyle='--', alpha=0.4)
        ax.set_xlabel('Avg VADER Compound Score')
        ax.set_title('Sentiment by Topic', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.subheader("Review Explorer")
    filter_sentiment = st.selectbox("Filter by sentiment", ["All"] + rev['SentimentLabel'].unique().tolist())
    filter_topic     = st.selectbox("Filter by topic",     ["All"] + rev['Topic'].unique().tolist())

    filtered = rev.copy()
    if filter_sentiment != "All":
        filtered = filtered[filtered['SentimentLabel'] == filter_sentiment]
    if filter_topic != "All":
        filtered = filtered[filtered['Topic'] == filter_topic]

    st.dataframe(
        filtered[['ReviewText','SentimentLabel','Topic','CompoundScore','StarRating']].head(20),
        use_container_width=True
    )
