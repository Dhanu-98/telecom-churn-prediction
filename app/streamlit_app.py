import joblib, numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import shap
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve
)

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# ---------- Paths (relative, portable) ----------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
MODEL_PATH = ROOT / "saved_models" / "best_pipe.pkl"
DATA_PATH  = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------- Load assets ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

best_pipe = load_model()

try:
    pre = best_pipe.named_steps["preprocess"]
    clf = best_pipe.named_steps["clf"]
except Exception:
    st.error("Pipeline must have steps 'preprocess' and 'clf'. Retrain with train.py.")
    st.stop()

@st.cache_data
def load_base():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None

base_df = load_base()

def raw_schema():
    if base_df is not None:
        return base_df.drop(columns=["Churn"], errors="ignore").columns.tolist()
    return [
        "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
    ]
RAW_COLS = raw_schema()

def retention_tip(row: pd.Series) -> str:
    tips = []
    if row.get("Contract") == "Month-to-month": tips.append("Offer 12-month plan with discount")
    if float(row.get("tenure", 0)) < 6 and float(row.get("MonthlyCharges", 0)) > 80: tips.append("Give 10% off for 3 months")
    if row.get("TechSupport") == "No": tips.append("Bundle free TechSupport trial")
    return "; ".join(tips) if tips else "Proactive care call"

def predict_df(df_raw: pd.DataFrame):
    prob = best_pipe.predict_proba(df_raw)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob

def feature_names():
    ohe = pre.named_transformers_['cat']
    cat_cols = pre.transformers_[0][2]
    num_cols = pre.transformers_[1][2]
    return np.concatenate([ohe.get_feature_names_out(cat_cols), num_cols])

def shap_explain(df_raw: pd.DataFrame):
    Xp = pre.transform(df_raw)
    try:
        expl = shap.TreeExplainer(clf)
        sv = expl(Xp)
        return sv, Xp
    except Exception as e:
        return None, Xp

# ---------- UI ----------
st.sidebar.title("Telecom Churn App")
st.sidebar.write(f"Model file: `{MODEL_PATH.name}`")

tab1, tab2, tab3 = st.tabs(["🔮 Single Predict", "📦 Batch Scoring", "📊 KPIs / Insights"])

with tab1:
    st.header("Single Customer Prediction")
    with st.form("f1"):
        c = st.columns(3)
        def sel(i, label, opts): return c[i].selectbox(label, opts)
        def num(i, label, v=0.0, step=1.0, lo=0.0, hi=10000.0): return c[i].number_input(label, value=v, step=step, min_value=lo, max_value=hi)
        def text(i, label, v=""): return c[i].text_input(label, value=v)

        def uniq(col, default):
            if base_df is not None and col in base_df.columns:
                return sorted([str(x) for x in base_df[col].dropna().unique().tolist()])
            return default

        gender = sel(0,"gender",uniq("gender",["Male","Female"]))
        senior = num(1,"SeniorCitizen (0/1)",v=0,step=1,lo=0,hi=1)
        partner = sel(2,"Partner",uniq("Partner",["Yes","No"]))
        depend = sel(0,"Dependents",uniq("Dependents",["Yes","No"]))
        tenure = num(1,"tenure (months)",v=12,step=1,lo=0,hi=100)
        phone  = sel(2,"PhoneService",uniq("PhoneService",["Yes","No"]))
        multi  = sel(0,"MultipleLines",uniq("MultipleLines",["Yes","No","No phone service"]))
        inet   = sel(1,"InternetService",uniq("InternetService",["DSL","Fiber optic","No"]))
        onsec  = sel(2,"OnlineSecurity",uniq("OnlineSecurity",["Yes","No","No internet service"]))
        onbak  = sel(0,"OnlineBackup",uniq("OnlineBackup",["Yes","No","No internet service"]))
        devp   = sel(1,"DeviceProtection",uniq("DeviceProtection",["Yes","No","No internet service"]))
        tech   = sel(2,"TechSupport",uniq("TechSupport",["Yes","No","No internet service"]))
        stv    = sel(0,"StreamingTV",uniq("StreamingTV",["Yes","No","No internet service"]))
        smv    = sel(1,"StreamingMovies",uniq("StreamingMovies",["Yes","No","No internet service"]))
        contract  = sel(2,"Contract",uniq("Contract",["Month-to-month","One year","Two year"]))
        paperless = sel(0,"PaperlessBilling",uniq("PaperlessBilling",["Yes","No"]))
        paym   = sel(1,"PaymentMethod",uniq("PaymentMethod",[
            "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]))
        monthly = num(2,"MonthlyCharges",v=70.0,step=0.1,lo=0,hi=1000)
        tot = text(0,"TotalCharges (blank = tenure*Monthly)","")
        cust_id = text(1,"customerID (optional)","NEW-USER-001")

        ok = st.form_submit_button("Predict")

    if ok:
        total = tot if tot.strip() else str(round(tenure*monthly,2))
        row = pd.DataFrame([{
            "customerID": cust_id, "gender": gender, "SeniorCitizen": int(senior),
            "Partner": partner, "Dependents": depend, "tenure": int(tenure),
            "PhoneService": phone, "MultipleLines": multi, "InternetService": inet,
            "OnlineSecurity": onsec, "OnlineBackup": onbak, "DeviceProtection": devp,
            "TechSupport": tech, "StreamingTV": stv, "StreamingMovies": smv,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": paym,
            "MonthlyCharges": float(monthly), "TotalCharges": total
        }])[RAW_COLS]

        pred, prob = predict_df(row)
        st.success(f"**{'Churn' if pred[0]==1 else 'No Churn'}**  |  Probability: **{prob[0]*100:.1f}%**")
        st.info(f"Suggested action: {retention_tip(row.iloc[0])}")

        # SHAP explanation
        try:
            shap_values, _ = shap_explain(row)
            if shap_values is not None:
                names = feature_names()
                st.subheader("Top reasons (SHAP)")
                fig = shap.plots._waterfall.waterfall_legacy(shap_values[0], feature_names=names, max_display=12, show=False)
                st.pyplot(fig)
            else:
                st.caption("SHAP not available for this model type.")
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

with tab2:
    st.header("Batch Scoring (CSV)")
    if base_df is not None:
        st.caption("Expected columns example (first 3 rows, without Churn):")
        st.dataframe(base_df.drop(columns=["Churn"], errors="ignore").head(3))
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_in = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df_in = df_in[RAW_COLS]
            preds, probs = predict_df(df_in)
            out = df_in.copy()
            out["Churn_Pred"] = preds
            out["Churn_Prob"] = probs
            out["Action"] = out.apply(retention_tip, axis=1)
            st.success("Done.")
            st.dataframe(out.head(20))
            st.download_button("Download Results", out.to_csv(index=False).encode(), "churn_scored.csv")

with tab3:
    st.header("KPIs / Insights")
    if base_df is None or "Churn" not in base_df.columns:
        st.info("Dataset with 'Churn' not found for KPI demo.")
    else:
        X = base_df.drop("Churn", axis=1)
        y = base_df["Churn"].map({"No":0,"Yes":1})
        preds, probs = predict_df(X)
        acc = accuracy_score(y, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        auc = roc_auc_score(y, probs)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{auc:.3f}")

        cm = confusion_matrix(y, preds)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i,j), z in np.ndenumerate(cm): ax.text(j,i,str(z),ha='center',va='center')
        st.pyplot(fig)

        fpr, tpr, _ = roc_curve(y, probs)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}"); ax2.plot([0,1],[0,1],'--')
        ax2.set_title("ROC Curve"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()
        st.pyplot(fig2)

st.caption("© Final-Year Project — Telecom Customer Churn")
