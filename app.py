
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import ks_2samp

from data_generator import (
    generate_reference_data,
    generate_live_data
)

from drift_monitor import run_drift_detection


# ------------------------------------
# Streamlit configuration
# ------------------------------------

st.set_page_config(
    page_title="Evidently AI Drift Monitoring",
    layout="wide"
)

st.title("Real-Time Data Drift Monitoring")
st.caption(
    "Bank Loan Applications | Evidently AI + Streamlit"
)


# ------------------------------------
# Initialize historical dataset
# ------------------------------------

if "reference_data" not in st.session_state:
    st.session_state.reference_data = (
        generate_reference_data()
    )

if "current_data" not in st.session_state:
    st.session_state.current_data = pd.DataFrame(
        columns=["age", "income", "credit_score"]
    )

if "batch_count" not in st.session_state:
    st.session_state.batch_count = 0


reference = st.session_state.reference_data


# ------------------------------------
# User controls
# ------------------------------------

st.sidebar.header("Simulation Controls")

drift_enabled = st.sidebar.toggle(
    "Introduce Data Drift",
    value=False
)

monitoring_enabled = st.sidebar.toggle(
    "Enable Live Monitoring",
    value=True
)

if st.sidebar.button("Reset Monitoring"):
    st.session_state.current_data = (
        st.session_state.current_data.iloc[0:0].copy()
    )
    st.session_state.batch_count = 0
    st.rerun()


st.sidebar.info(
    "50 new customer records arrive every 5 seconds."
)


# ------------------------------------
# Live monitoring function
# ------------------------------------

@st.fragment(run_every="5s")
def live_monitor():

    if monitoring_enabled:

        # Generate 50 incoming customers
        new_batch = generate_live_data(
            drift=drift_enabled,
            batch_size=50
        )

        # Append to the current dataset
        st.session_state.current_data = pd.concat(
            [
                st.session_state.current_data,
                new_batch
            ],
            ignore_index=True
        ).tail(500)

        st.session_state.batch_count += 1

    current = st.session_state.current_data

    # ------------------------------------
    # Live metrics
    # ------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Reference Customers",
        len(reference)
    )

    col2.metric(
        "Current Window",
        len(current)
    )

    col3.metric(
        "Batches Processed",
        st.session_state.batch_count
    )

    if len(current) < 100:
        st.warning(
            "Collecting data. Waiting for 100 records."
        )
        return

    # ------------------------------------
    # Run Evidently AI
    # ------------------------------------

    st.subheader("Evidently AI Drift Evaluation")

    snapshot = run_drift_detection(
        reference,
        current
    )

    os.makedirs("reports", exist_ok=True)

    snapshot.save_html(
        "reports/latest_drift_report.html"
    )

    st.success(
        "Evidently evaluation completed successfully"
    )

    # ------------------------------------
    # Feature-level KS diagnostics
    # ------------------------------------

    st.subheader("Live Statistical Drift Alerts")

    results = []

    for feature in reference.columns:

        ref_values = reference[feature].to_numpy(
            dtype=float
        )

        cur_values = current[feature].to_numpy(
            dtype=float
        )

        statistic, p_value = ks_2samp(
            ref_values,
            cur_values
        )

        drift_detected = p_value < 0.05

        results.append({
            "Feature": feature,
            "Reference Mean": round(
                ref_values.mean(), 2
            ),
            "Current Mean": round(
                cur_values.mean(), 2
            ),
            "KS Statistic": round(
                statistic, 4
            ),
            "P Value": round(
                p_value, 6
            ),
            "Drift Detected": drift_detected
        })

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    drift_count = int(
        results_df["Drift Detected"].sum()
    )

    if drift_count > 0:
        st.error(
            f"DRIFT ALERT: {drift_count} features "
            "show statistically significant changes"
        )
    else:
        st.success(
            "No statistically significant drift detected"
        )

    # ------------------------------------
    # Distribution visualization
    # ------------------------------------

    st.subheader("Reference vs Current Distribution")

    feature = st.selectbox(
        "Select Feature",
        list(reference.columns)
    )

    fig = go.Figure()

    fig.add_histogram(
        x=reference[feature],
        name="Historical Data",
        opacity=0.6,
        histnorm="probability density"
    )

    fig.add_histogram(
        x=current[feature],
        name="Live Customer Data",
        opacity=0.6,
        histnorm="probability density"
    )

    fig.update_layout(
        title=f"{feature} Distribution Comparison",
        barmode="overlay",
        xaxis_title=feature,
        yaxis_title="Density"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------
    # Download Evidently HTML report
    # ------------------------------------

    with open(
        "reports/latest_drift_report.html",
        "rb"
    ) as file:

        st.download_button(
            "Download Evidently Drift Report",
            data=file.read(),
            file_name="evidently_drift_report.html",
            mime="text/html"
        )


live_monitor()
