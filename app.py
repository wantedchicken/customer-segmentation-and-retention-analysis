import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation & Retention Dashboard",
    layout="wide"
)

st.title("Customer Segmentation & Retention Intelligence")
st.caption("An interactive customer intelligence tool for segmentation, retention analysis, and churn risk targeting.")
st.caption("RFM Segmentation • Cohort Retention • Churn Risk Scoring (Online Retail)")


# -----------------------------
# Helpers (Data + Features)
# -----------------------------
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    """Load + clean the Online Retail dataset."""
    df = pd.read_excel(path)

    # Basic cleaning (standard for this dataset)
    df = df.dropna(subset=["CustomerID"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # remove cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    rfm = (
        df.groupby("CustomerID")
          .agg(
              Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
              Frequency=("InvoiceNo", "nunique"),
              Monetary=("TotalPrice", "sum"),
          )
          .reset_index()
    )
    return rfm


def rfm_score(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    out["R_Score"] = pd.qcut(out["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_Score"] = pd.qcut(out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_Score"] = pd.qcut(out["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_Score"] = out["R_Score"].astype(str) + out["F_Score"].astype(str) + out["M_Score"].astype(str)
    return out


def assign_segment(row) -> str:
    if row["R_Score"] >= 4 and row["F_Score"] >= 4:
        return "Champions"
    elif row["F_Score"] >= 4:
        return "Loyal Customers"
    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "At Risk"
    elif row["R_Score"] >= 4:
        return "New Customers"
    else:
        return "Others"


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["InvoiceMonth"] = tmp["InvoiceDate"].dt.to_period("M")
    tmp["CohortMonth"] = tmp.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    tmp["CohortIndex"] = (tmp["InvoiceMonth"] - tmp["CohortMonth"]).apply(lambda x: x.n) + 1

    cohort = (
        tmp.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
           .nunique()
           .reset_index()
    )
    pivot = cohort.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")
    retention = pivot.divide(pivot.iloc[:, 0], axis=0)
    return retention


def churn_features_and_labels(df: pd.DataFrame, churn_window_days: int = 90):
    """
    Define churn as: no purchase in the next churn_window_days after cutoff date.
    Build features from history up to cutoff date.
    """
    cutoff_date = df["InvoiceDate"].max() - pd.Timedelta(days=churn_window_days)

    df_hist = df[df["InvoiceDate"] <= cutoff_date].copy()
    df_future = df[(df["InvoiceDate"] > cutoff_date) &
                   (df["InvoiceDate"] <= cutoff_date + pd.Timedelta(days=churn_window_days))].copy()

    snapshot_date = cutoff_date + pd.Timedelta(days=1)

    feats = (
        df_hist.groupby("CustomerID")
            .agg(
                Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("TotalPrice", "sum"),
                AvgOrderValue=("TotalPrice", "mean"),
                UniqueProducts=("StockCode", "nunique"),
                ActiveDays=("InvoiceDate", lambda x: x.dt.date.nunique()),
            )
            .reset_index()
    )

    future_buyers = set(df_future["CustomerID"].unique())
    feats["Churn"] = (~feats["CustomerID"].isin(future_buyers)).astype(int)
    return feats, cutoff_date


@st.cache_resource
def train_churn_model_cached(feats: pd.DataFrame):
    """
    Train a Random Forest churn model and return:
    (model, X, y, (roc_auc, pr_auc))
    Cached so the dashboard stays fast.
    """
    X = feats[["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts", "ActiveDays"]]
    y = feats["Churn"]

    # Guardrail
    if len(feats) < 200 or y.nunique() < 2:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    return rf, X, y, (roc, pr)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

data_path = st.sidebar.text_input(
    "Dataset path (xlsx)",
    value="data/Online Retail.xlsx"  # correct default when app.py is in project root
)

churn_window = st.sidebar.slider("Churn window (days)", 30, 180, 90, step=15)
k_clusters = st.sidebar.slider("K-Means clusters (k)", 2, 8, 4, step=1)

st.sidebar.divider()
show_raw = st.sidebar.checkbox("Show sample raw data", value=False)

if st.sidebar.button("Clear cache (reload data/model)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# -----------------------------
# Load data
# -----------------------------
try:
    df = load_raw_data(data_path)
except Exception as e:
    st.error("Could not load dataset. Please check the path in the sidebar.")
    st.code(str(e))
    st.stop()

snapshot_date_full = df["InvoiceDate"].max() + pd.Timedelta(days=1)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segmentation", "Retention", "Churn Prediction"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions (cleaned)", f"{len(df):,}")
    c2.metric("Customers", f"{df['CustomerID'].nunique():,}")
    c3.metric("Countries", f"{df['Country'].nunique():,}")
    c4.metric("Revenue(£)", f"{df['TotalPrice'].sum():,.0f}")

    if show_raw:
        st.subheader("Sample data")
        st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Executive Summary")
    st.markdown("""
- **High-value customers**: Champions / Loyal segments drive disproportionate revenue.
- **High-risk customers**: At-Risk + one-time buyers show high churn likelihood, especially after early inactivity.
- **Action**: Use churn risk scores to prioritize win-back campaigns and protect high-value customers with loyalty incentives.
""")

    st.subheader("Top Countries by Revenue")
    top_ctry = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_ctry)


with tab2:
    st.subheader("RFM Segmentation (Business Personas)")

    rfm = build_rfm(df, snapshot_date_full)
    rfm = rfm_score(rfm)
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Segment Distribution")
        seg_counts = rfm["Segment"].value_counts()
        st.bar_chart(seg_counts)

    with right:
        st.markdown("### Revenue by Segment")
        seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
        st.bar_chart(seg_rev)

    st.markdown("### K-Means (ML Segmentation)")
    X_km = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_km)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    rfm["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    cs1, cs2 = st.columns([1, 1])
    with cs1:
        st.markdown("**Cluster sizes**")
        st.dataframe(
            rfm["KMeans_Cluster"].value_counts().rename("count").to_frame(),
            use_container_width=True
        )

    with cs2:
        st.markdown("**Cluster summary (means)**")
        cluster_summary = rfm.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        st.dataframe(cluster_summary, use_container_width=True)

    st.markdown("### Customer Lookup")
    cust_id = st.number_input("Enter CustomerID", min_value=0, step=1, value=0)
    if cust_id != 0:
        row = rfm[rfm["CustomerID"] == cust_id]
        if row.empty:
            st.warning("CustomerID not found in RFM table.")
        else:
            st.dataframe(row, use_container_width=True)


with tab3:
    st.subheader("Cohort Retention Analysis")
    retention = cohort_retention(df)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(retention, annot=False, cmap="Blues", ax=ax)
    ax.set_title("Retention Heatmap (Cohorts)")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    st.pyplot(fig)

    st.markdown("### Retention Table (first rows)")
    st.dataframe(retention.head(10), use_container_width=True)


with tab4:
    st.subheader("Churn Prediction (Supervised ML)")

    feats, cutoff_date = churn_features_and_labels(df, churn_window_days=churn_window)
    churn_rate = feats["Churn"].mean()

    st.info(f"Churn definition: no purchase in the next **{churn_window} days** after cutoff date.")
    st.write(
        f"Cutoff date: **{cutoff_date.date()}** • "
        f"Customers labeled: **{len(feats):,}** • "
        f"Churn rate: **{churn_rate:.1%}**"
    )

    model, X_churn, y_churn, scores = train_churn_model_cached(feats)

    if model is None:
        st.warning("Not enough data to train a churn model (or only one class present). Try a different churn window.")
        st.stop()

    roc, pr = scores
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC", f"{roc:.3f}")
    m2.metric("PR-AUC", f"{pr:.3f}")

    # Score all labeled customers
    feats = feats.copy()
    feats["Churn_Prob"] = model.predict_proba(X_churn)[:, 1]

    st.subheader("Recommended Actions")
    st.markdown("""
- **High-risk customers**: win-back email + limited-time offer; prioritize highest churn probability.
- **New customers**: onboarding + first-repeat incentive within 30 days.
- **Champions/Loyal**: loyalty perks, early access, and personalized bundles to protect revenue.
""")

    st.markdown("### Retention Target List")
    min_prob = st.slider("Minimum churn probability", 0.0, 1.0, 0.7, 0.05)
    top_n = st.slider("Top N customers", 10, 300, 50, step=10)

    filtered = feats[feats["Churn_Prob"] >= min_prob].sort_values("Churn_Prob", ascending=False)
    st.write(f"Customers above threshold: **{len(filtered):,}**")

    top_risk = filtered.head(top_n)

    st.dataframe(
        top_risk[["CustomerID", "Churn_Prob", "Recency", "Frequency", "Monetary", "ActiveDays"]].reset_index(drop=True),
        use_container_width=True
    )

    csv = top_risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Target List (CSV)",
        data=csv,
        file_name=f"target_list_{churn_window}d_minprob_{min_prob:.2f}.csv",
        mime="text/csv"
    )

    st.markdown("### Feature Importance (Random Forest)")
    fi = pd.Series(model.feature_importances_, index=X_churn.columns).sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fi.plot(kind="bar", ax=ax2)
    ax2.set_title("Feature Importance")
    ax2.set_ylabel("Importance")
    st.pyplot(fig2)
