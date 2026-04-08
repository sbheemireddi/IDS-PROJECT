import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════════════════
#  Load YOUR trained model + selected features
#  Place both .pkl files in the same folder as this script.
#  If they are on Google Drive, update the paths below.
# ══════════════════════════════════════════════════════════

MODEL_PATH    = "ids_ensemble_model.pkl"
FEATURES_PATH = "selected_features.pkl"

@st.cache_resource
def load_artifacts():
    errors = []
    if not os.path.exists(MODEL_PATH):
        errors.append(f"❌ Not found: `{MODEL_PATH}`")
    if not os.path.exists(FEATURES_PATH):
        errors.append(f"❌ Not found: `{FEATURES_PATH}`")
    if errors:
        return None, None, errors
    model             = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURES_PATH)   # list / array of feature names
    return model, list(selected_features), []

model, selected_features, load_errors = load_artifacts()

# ══════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title="IDS Dashboard", page_icon="🔐", layout="wide")
st.title("🔐 Industrial IoT Intrusion Detection System")
st.subheader("Real-Time Network Intrusion Detection Dashboard")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Model Info")
    if load_errors:
        for e in load_errors:
            st.error(e)
        st.info(
            "Place **ids_ensemble_model.pkl** and **selected_features.pkl** "
            "in the same directory as this script, then restart the app."
        )
    else:
        st.success("✅ ids_ensemble_model.pkl loaded")
        st.success("✅ selected_features.pkl loaded")
        st.write(f"**Number of features:** {len(selected_features)}")
        with st.expander("Show feature names"):
            for f in selected_features:
                st.write(f"• {f}")

    st.markdown("---")
    st.subheader("⚙️ Simulation Settings")
    n_steps   = st.slider("Flows to simulate", 10, 100, 30)
    delay_sec = st.slider("Delay between flows (s)", 0.1, 2.0, 0.5, step=0.1)

# ══════════════════════════════════════════════════════════
#  Feature value profiles (covers most CICIDS / NSL-KDD
#  style column names).  Unknown features fall back to a
#  generic low/high range.
# ══════════════════════════════════════════════════════════

_PROFILES = {
    # feature_name                  : (normal_low, normal_high, attack_low, attack_high)
    "Flow Duration"                 : (100,   2000,   8000, 20000),
    "Total Fwd Packets"             : (5,       40,    200,   500),
    "Total Backward Packets"        : (5,       40,    200,   500),
    "Total Length of Fwd Packets"   : (100,    600,    900,  1500),
    "Total Length of Bwd Packets"   : (100,    600,    900,  1500),
    "Fwd Packet Length Max"         : (100,    600,    900,  1500),
    "Fwd Packet Length Min"         : (20,     100,    400,   900),
    "Fwd Packet Length Mean"        : (50,     300,    600,  1200),
    "Fwd Packet Length Std"         : (10,     100,    200,   600),
    "Bwd Packet Length Max"         : (100,    600,    900,  1500),
    "Bwd Packet Length Min"         : (20,     100,    400,   900),
    "Bwd Packet Length Mean"        : (50,     300,    600,  1200),
    "Bwd Packet Length Std"         : (10,     100,    200,   600),
    "Flow Bytes/s"                  : (1000, 50000, 500000,2000000),
    "Flow Packets/s"                : (10,   1000,   5000,  50000),
    "Flow IAT Mean"                 : (1000, 50000,     50,    500),
    "Flow IAT Std"                  : (500,  20000,     10,    200),
    "Flow IAT Max"                  : (5000,100000,    100,   2000),
    "Flow IAT Min"                  : (0,    1000,       0,     50),
    "Fwd IAT Total"                 : (1000, 50000,     50,    500),
    "Fwd IAT Mean"                  : (500,  20000,     20,    200),
    "Fwd IAT Std"                   : (200,  10000,     10,    100),
    "Fwd IAT Max"                   : (2000, 80000,     50,   1000),
    "Fwd IAT Min"                   : (0,    1000,       0,     20),
    "Bwd IAT Total"                 : (1000, 50000,     50,    500),
    "Bwd IAT Mean"                  : (500,  20000,     20,    200),
    "Bwd IAT Std"                   : (200,  10000,     10,    100),
    "Bwd IAT Max"                   : (2000, 80000,     50,   1000),
    "Bwd IAT Min"                   : (0,    1000,       0,     20),
    "Fwd PSH Flags"                 : (0,       1,      0,      1),
    "Bwd PSH Flags"                 : (0,       1,      0,      1),
    "Fwd URG Flags"                 : (0,       1,      0,      1),
    "Bwd URG Flags"                 : (0,       1,      0,      1),
    "Fwd Header Length"             : (20,    100,    200,    600),
    "Bwd Header Length"             : (20,    100,    200,    600),
    "Fwd Packets/s"                 : (5,     500,   2000,  20000),
    "Bwd Packets/s"                 : (5,     500,   2000,  20000),
    "Min Packet Length"             : (20,    200,    400,    900),
    "Max Packet Length"             : (100,   600,    900,   1500),
    "Packet Length Mean"            : (50,    300,    600,   1200),
    "Packet Length Std"             : (20,    150,    300,    700),
    "Packet Length Variance"        : (400, 22500,  90000, 490000),
    "FIN Flag Count"                : (0,      2,      0,      5),
    "SYN Flag Count"                : (0,      2,      5,     50),
    "RST Flag Count"                : (0,      1,      0,     10),
    "PSH Flag Count"                : (0,      5,      0,     20),
    "ACK Flag Count"                : (0,     10,     10,    100),
    "URG Flag Count"                : (0,      1,      0,      5),
    "CWE Flag Count"                : (0,      1,      0,      2),
    "ECE Flag Count"                : (0,      1,      0,      2),
    "Down/Up Ratio"                 : (0.5,   2.0,    5.0,   20.0),
    "Average Packet Size"           : (50,    300,    600,   1200),
    "Avg Fwd Segment Size"          : (50,    300,    600,   1200),
    "Avg Bwd Segment Size"          : (50,    300,    600,   1200),
    "Fwd Avg Bytes/Bulk"            : (0,     100,    500,   2000),
    "Fwd Avg Packets/Bulk"          : (0,      10,     50,    200),
    "Fwd Avg Bulk Rate"             : (0,    1000,   5000,  50000),
    "Bwd Avg Bytes/Bulk"            : (0,     100,    500,   2000),
    "Bwd Avg Packets/Bulk"          : (0,      10,     50,    200),
    "Bwd Avg Bulk Rate"             : (0,    1000,   5000,  50000),
    "Subflow Fwd Packets"           : (5,      40,    200,    500),
    "Subflow Fwd Bytes"             : (100,   600,    900,   1500),
    "Subflow Bwd Packets"           : (5,      40,    200,    500),
    "Subflow Bwd Bytes"             : (100,   600,    900,   1500),
    "Init_Win_bytes_forward"        : (8192,65535,      0,   1024),
    "Init_Win_bytes_backward"       : (8192,65535,      0,   1024),
    "act_data_pkt_fwd"              : (1,      20,    100,    500),
    "min_seg_size_forward"          : (20,     40,      0,     20),
    "Active Mean"                   : (1000, 50000,    50,    500),
    "Active Std"                    : (500,  20000,    10,    200),
    "Active Max"                    : (5000,100000,   100,   2000),
    "Active Min"                    : (500,  10000,    10,    200),
    "Idle Mean"                     : (1000, 80000,    50,   1000),
    "Idle Std"                      : (500,  30000,    10,    500),
    "Idle Max"                      : (5000,200000,   200,   5000),
    "Idle Min"                      : (500,  50000,    10,   1000),
}

def generate_flow(features_list):
    """
    Simulate one network flow whose column names match
    exactly what YOUR model was trained on (selected_features).
    """
    is_normal      = np.random.rand() < 0.7
    simulated_type = "Normal" if is_normal else "Attack"

    row = {}
    for feat in features_list:
        if feat in _PROFILES:
            nl, nh, al, ah = _PROFILES[feat]
            row[feat] = np.random.uniform(nl, nh) if is_normal else np.random.uniform(al, ah)
        else:
            # Fallback for any feature name not in the profile dict
            row[feat] = np.random.uniform(0, 50) if is_normal else np.random.uniform(100, 500)

    return pd.DataFrame([row]), simulated_type

# ══════════════════════════════════════════════════════════
#  Session state (keeps counts across reruns)
# ══════════════════════════════════════════════════════════

for key, default in [("attack_count", 0), ("normal_count", 0), ("data_log", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Live metric cards ─────────────────────────────────────
col_n, col_a, col_t = st.columns(3)
metric_normal = col_n.empty()
metric_attack = col_a.empty()
metric_total  = col_t.empty()

def refresh_metrics():
    n = st.session_state.normal_count
    a = st.session_state.attack_count
    metric_normal.metric("✅ Normal Flows",  n)
    metric_attack.metric("🚨 Attack Flows",  a)
    metric_total.metric( "📊 Total Flows",   n + a)

refresh_metrics()

# ── Placeholders ─────────────────────────────────────────
table_placeholder = st.empty()
chart_placeholder = st.empty()

# ── Reset button ──────────────────────────────────────────
if st.button("🔄 Reset Dashboard"):
    st.session_state.attack_count = 0
    st.session_state.normal_count = 0
    st.session_state.data_log     = []
    refresh_metrics()
    table_placeholder.empty()
    chart_placeholder.empty()

# ══════════════════════════════════════════════════════════
#  Simulation
# ══════════════════════════════════════════════════════════

if st.button("▶️ Start Real-Time Simulation"):

    # ── Guard: model must be loaded ───────────────────────
    if load_errors:
        st.error(
            "Cannot start — model files are missing. "
            "See the sidebar for instructions."
        )
        st.stop()

    progress_bar = st.progress(0)
    status_box   = st.empty()

    for step in range(n_steps):

        # 1. Generate a flow with YOUR feature columns
        feature_df, simulated_type = generate_flow(selected_features)

        # 2. YOUR ensemble model predicts
        raw_pred = model.predict(feature_df)[0]

        # Handle numeric (0/1) or string labels
        if isinstance(raw_pred, (int, np.integer)):
            prediction = "Attack" if int(raw_pred) == 1 else "Normal"
        else:
            prediction = str(raw_pred)

        # 3. Confidence (if model supports predict_proba)
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba      = model.predict_proba(feature_df)[0]
                confidence = round(float(max(proba)) * 100, 1)
            except Exception:
                pass

        # 4. Update counters
        if prediction == "Attack":
            st.session_state.attack_count += 1
        else:
            st.session_state.normal_count += 1
        refresh_metrics()

        # 5. Build display row (first 3 selected features + labels)
        display_cols = selected_features[:3]
        row = {c: round(float(feature_df[c].iloc[0]), 2) for c in display_cols}
        row["Simulated Traffic"] = simulated_type
        row["Model Prediction"]  = prediction
        if confidence is not None:
            row["Confidence (%)"] = confidence
        st.session_state.data_log.append(row)

        # 6. Colour-coded table (last 10 rows)
        df_log = pd.DataFrame(st.session_state.data_log)

        def colour_pred(val):
            if val == "Attack":
                return "background-color:#ffcccc; color:#900"
            if val == "Normal":
                return "background-color:#ccffcc; color:#060"
            return ""

        table_placeholder.dataframe(
            df_log.tail(10).style.applymap(colour_pred, subset=["Model Prediction"]),
            use_container_width=True
        )

        # 7. Two-panel chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Left – cumulative bar chart
        counts  = [st.session_state.normal_count, st.session_state.attack_count]
        clr     = ["#4CAF50", "#F44336"]
        ax1.bar(["Normal", "Attack"], counts, color=clr, edgecolor="black", width=0.5)
        for idx, v in enumerate(counts):
            ax1.text(idx, v + 0.3, str(v), ha="center", fontweight="bold", fontsize=12)
        ax1.set_title("Cumulative Detection", fontsize=13)
        ax1.set_ylabel("Number of Flows")
        ax1.set_ylim(0, max(counts) + 5)

        # Right – recent-10 prediction timeline
        recent = df_log.tail(10)["Model Prediction"].tolist()
        bar_clr = ["#F44336" if p == "Attack" else "#4CAF50" for p in recent]
        ax2.bar(range(len(recent)), [1] * len(recent), color=bar_clr, edgecolor="white")
        ax2.set_title("Last 10 Flow Predictions", fontsize=13)
        ax2.set_xticks(range(len(recent)))
        ax2.set_xticklabels([p[0] for p in recent])   # A / N labels
        ax2.set_yticks([])
        ax2.legend(
            handles=[
                plt.Rectangle((0,0),1,1, color="#4CAF50", label="Normal"),
                plt.Rectangle((0,0),1,1, color="#F44336", label="Attack"),
            ], loc="upper right", fontsize=9
        )

        plt.tight_layout()
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        # 8. Progress & status
        progress_bar.progress((step + 1) / n_steps)
        emoji  = "🚨 ATTACK DETECTED" if prediction == "Attack" else "✅ Normal traffic"
        conf_s = f" (confidence: {confidence}%)" if confidence else ""
        status_box.markdown(f"**Flow {step + 1}/{n_steps}** → {emoji}{conf_s}")

        time.sleep(delay_sec)

    status_box.success("✅ Simulation complete!")
    progress_bar.empty()

# ══════════════════════════════════════════════════════════
#  Footer summary
# ══════════════════════════════════════════════════════════

st.markdown("---")
total = st.session_state.normal_count + st.session_state.attack_count
if total > 0:
    pct = round(st.session_state.attack_count / total * 100, 1)
    st.info(
        f"**Session summary:** {total} flows analysed — "
        f"🟢 {st.session_state.normal_count} Normal  |  "
        f"🔴 {st.session_state.attack_count} Attacks ({pct}%)"
    )
