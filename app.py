# =========================================================
# STREAMLIT APP: HOSPITAL STAY RISK INTELLIGENCE
# Diabetes 130-US Hospitals (1999-2008) | UCI ML Repository
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from sklearn.pipeline import Pipeline as SKPipeline

pd.set_option("styler.render.max_elements", 10_000_000)

st.set_page_config(
    page_title="Hospital Stay Risk Intelligence",
    layout="wide",
    page_icon="🏥"
)

# =========================================================
# COLORS
# =========================================================
COLORS = {
    "navy":      "#1B3A6B",
    "steel":     "#2E6DA4",
    "mid":       "#4A90C4",
    "lightblue": "#A8C8E8",
    "pale":      "#EEF4FA",
    "slate":     "#6B7C93",
    "bg":        "#F4F7FB",
    "white":     "#FFFFFF",
    "border":    "#D0DBE8",
    "red":       "#C0392B",
    "redbg":     "#FEF0EE",
    "amber":     "#D4870A",
    "amberbg":   "#FEF9EE",
    "green":     "#1A7A4A",
    "greenbg":   "#EEF9F3",
}
risk_colors = [COLORS["lightblue"], COLORS["steel"], COLORS["navy"]]

sns.set_style("white")
plt.rcParams.update({
    "axes.grid": False, "font.size": 13, "axes.titlesize": 15,
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# =========================================================
# CSS — removed white chart body tile, kept only navy title bar
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* {{ font-family: 'Inter', sans-serif !important; box-sizing: border-box; }}
.stApp {{ background: {COLORS["bg"]}; }}
header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding: 0rem 2rem 3rem 2rem; max-width: 1600px; }}

.hdr {{
    background: linear-gradient(135deg, {COLORS["navy"]} 0%, {COLORS["steel"]} 60%, {COLORS["mid"]} 100%);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(27,58,107,0.22);
    display: flex; align-items: center; gap: 32px; min-height: 160px;
}}
.hdr-text {{ flex: 1; }}
.hdr-title {{ color:#fff; font-size:34px; font-weight:900; letter-spacing:-0.5px; margin-bottom:8px; line-height:1.2; }}
.hdr-sub {{ color:#BDD5EE; font-size:14.5px; line-height:1.7; margin-bottom:16px; }}
.hdr-badges {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }}
.badge {{ background:rgba(255,255,255,0.13); color:#E8F2FC; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:600; border:1px solid rgba(255,255,255,0.22); }}
.badge-src {{ background:rgba(255,255,255,0.08); color:#A8C8E8; padding:5px 12px; border-radius:12px; font-size:11px; font-weight:500; border:1px solid rgba(255,255,255,0.15); display:inline-block; }}

.kcard {{ background:{COLORS["white"]}; border-radius:16px; padding:22px 16px; text-align:center; box-shadow:0 2px 14px rgba(27,58,107,0.08); border-top:4px solid {COLORS["navy"]}; height:120px; display:flex; flex-direction:column; justify-content:center; align-items:center; }}
.kcard-label {{ font-size:11px; font-weight:700; color:{COLORS["slate"]}; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }}
.kcard-value {{ font-size:26px; font-weight:800; color:{COLORS["navy"]}; line-height:1.1; }}
.kcard-sub {{ font-size:11px; color:{COLORS["slate"]}; margin-top:5px; }}

/* Section header — navy bar, white text, NO white body below */
.section-hdr {{ background:{COLORS["navy"]}; color:white; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1px; padding:8px 14px; border-radius:8px; margin-bottom:12px; display:block; }}

/* Chart title — navy bar only, no white body div */
.chart-title {{ background:{COLORS["navy"]}; color:white; font-size:13px; font-weight:700; padding:9px 16px; border-radius:10px; margin-bottom:8px; display:block; letter-spacing:0.3px; }}

.rcard-low {{ background:{COLORS["greenbg"]}; border-left:8px solid {COLORS["green"]}; border-radius:14px; padding:22px; text-align:center; font-size:22px; font-weight:800; color:{COLORS["green"]}; }}
.rcard-med {{ background:{COLORS["amberbg"]}; border-left:8px solid {COLORS["amber"]}; border-radius:14px; padding:22px; text-align:center; font-size:22px; font-weight:800; color:{COLORS["amber"]}; }}
.rcard-high {{ background:{COLORS["redbg"]}; border-left:8px solid {COLORS["red"]}; border-radius:14px; padding:22px; text-align:center; font-size:22px; font-weight:800; color:{COLORS["red"]}; }}

.ibox {{ background:{COLORS["white"]}; border-radius:14px; padding:18px 20px; border:1px solid {COLORS["border"]}; font-size:13.5px; line-height:1.7; box-shadow:0 1px 8px rgba(27,58,107,0.05); }}
.inbox-navy {{ background:{COLORS["pale"]}; border-left:5px solid {COLORS["navy"]}; border-radius:12px; padding:14px 18px; font-size:13.5px; color:{COLORS["navy"]}; line-height:1.7; margin-top:8px; }}
.inbox-steel {{ background:#EAF3FB; border-left:5px solid {COLORS["steel"]}; border-radius:12px; padding:14px 18px; font-size:13.5px; color:{COLORS["navy"]}; line-height:1.7; margin-top:8px; }}
.warn-box {{ background:{COLORS["amberbg"]}; border-left:5px solid {COLORS["amber"]}; border-radius:10px; padding:12px 16px; font-size:13px; color:#5a3800; margin-top:10px; }}
.ready-box {{ background:{COLORS["white"]}; border-radius:18px; padding:60px 40px; text-align:center; border:2px dashed {COLORS["border"]}; box-shadow:0 2px 12px rgba(27,58,107,0.05); }}

div[data-baseweb="tab-list"] {{ gap:6px; border-bottom:2px solid {COLORS["border"]} !important; }}
button[data-baseweb="tab"] {{ font-size:14px !important; font-weight:700 !important; padding:10px 22px !important; border-radius:10px 10px 0 0 !important; color:{COLORS["slate"]} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background:{COLORS["navy"]} !important; color:white !important; }}
button[data-baseweb="tab"]:hover {{ background:{COLORS["pale"]} !important; color:{COLORS["navy"]} !important; }}
hr {{ border-color:{COLORS["border"]}; margin:20px 0; }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE CHECK
# =========================================================
REQUIRED = [
    "diabetes_los_model.pkl",
    "diabetes_los_feature_columns.pkl",
    "diabetes_los_feature_template.pkl",
    "diabetes_los_threshold.pkl",
    "diabetes_diag_avg_los.pkl",
    "diabetes_dashboard_full.csv",
    "diabetes_triage_summary.csv",
    "diabetes_feature_importance.csv",
    "diabetes_diag_group_perf.csv",
]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error("Missing required files:")
    for f in missing: st.code(f)
    st.stop()

# =========================================================
# LOAD
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_los_model.pkl")

@st.cache_resource
def load_assets():
    return (
        joblib.load("diabetes_los_feature_columns.pkl"),
        joblib.load("diabetes_los_feature_template.pkl"),
        joblib.load("diabetes_los_threshold.pkl"),
        joblib.load("diabetes_diag_avg_los.pkl"),
    )

@st.cache_data
def load_data():
    return (
        pd.read_csv("diabetes_dashboard_full.csv"),
        pd.read_csv("diabetes_triage_summary.csv"),
        pd.read_csv("diabetes_feature_importance.csv"),
        pd.read_csv("diabetes_diag_group_perf.csv"),
    )

model                                           = load_model()
feature_cols, feat_tmpl, THRESHOLD, diag_map   = load_assets()
dashboard, triage_df, importance_df, diag_perf = load_data()

risk_order = ["Low Risk", "Medium Risk", "High Risk"]
triage_df["Risk_Level"] = pd.Categorical(
    triage_df["Risk_Level"], categories=risk_order, ordered=True
)
triage_df = triage_df.sort_values("Risk_Level").reset_index(drop=True)

IS_PIPELINE  = isinstance(model, SKPipeline)
NUMERIC_COLS = feat_tmpl.select_dtypes(include=[np.number]).columns.tolist()
CATEG_COLS   = feat_tmpl.select_dtypes(exclude=[np.number]).columns.tolist()

# Correct values from notebook output
F2_SCORE  = 0.6888
RECALL    = 0.6934

# =========================================================
# HELPERS
# =========================================================
def kpi(label, value, sub="", top_color=None):
    tc       = top_color or COLORS["navy"]
    sub_html = f'<div class="kcard-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kcard" style="border-top-color:{tc};">'
        f'<div class="kcard-label">{label}</div>'
        f'<div class="kcard-value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )

def section_hdr(title):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)

def chart_title(title):
    """Navy title bar only — no white body wrapper."""
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)

def assign_risk(prob):
    if prob >= THRESHOLD + 0.15:  return "High Risk"
    elif prob >= THRESHOLD:       return "Medium Risk"
    else:                         return "Low Risk"

def action_text(risk):
    return {
        "High Risk":   "Priority bed planning. Assign case manager immediately. Begin discharge coordination from Day 1.",
        "Medium Risk": "Enhanced monitoring. Review resource needs. Mid-admission discharge planning check.",
        "Low Risk":    "Standard admission workflow. No immediate escalation required."
    }[risk]

def clean(ax):
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

def label_v(ax, fmt="{:,.0f}", fontsize=11, pad=0.01):
    vals = [p.get_height() for p in ax.patches]
    mx   = max(vals) if vals else 1
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x()+p.get_width()/2, h+mx*pad,
                    fmt.format(h), ha="center", va="bottom",
                    fontweight="bold", fontsize=fontsize, color=COLORS["navy"])

def label_h(ax, fmt="{:.0f}", fontsize=11, pad=0.01):
    vals = [p.get_width() for p in ax.patches]
    mx   = max(vals) if vals else 1
    for p in ax.patches:
        w = p.get_width()
        if w > 0:
            ax.text(w+mx*pad, p.get_y()+p.get_height()/2,
                    fmt.format(w), va="center",
                    fontsize=fontsize, fontweight="bold", color=COLORS["navy"])

def bar3_colors(values):
    mx, mn = max(values), min(values)
    return [
        COLORS["navy"]      if v == mx else
        COLORS["lightblue"] if v == mn else
        COLORS["steel"]
        for v in values
    ]

def clean_feature_name(name):
    replacements = {
        "clinical_service_intensity": "Clinical Service Intensity",
        "medication_change_count":    "Medication Change Count",
        "diag_3_group_Diabetes":      "Diag 3: Diabetes",
        "diag_2_group_Diabetes":      "Diag 2: Diabetes",
        "diag_2_group_Circulatory":   "Diag 2: Circulatory",
        "diag_3_group_Circulatory":   "Diag 3: Circulatory",
        "diag_3_group_Skin":          "Diag 3: Skin",
        "diag_2_group_Skin":          "Diag 2: Skin",
        "max_glu_serum_>300":         "Max Glucose Serum >300",
        "max_glu_serum_>200":         "Max Glucose Serum >200",
        "gender_Female":              "Gender: Female",
        "gender_Male":                "Gender: Male",
        "insulin_No":                 "Insulin: No",
        "change_Ch":                  "Medication Changed",
        "num_medications":            "Num Medications",
        "number_diagnoses":           "Num Diagnoses",
        "age_midpoint":               "Age (Midpoint)",
        "diag_group_avg_los":         "Diag Group Median LOS",
        "prior_utilization":          "Prior Utilisation",
    }
    return replacements.get(name, name.replace("_"," ").title())

def calc_scenario(ext_tot, extra, avg_los_all, rate, cost_bed, rev_pt, years):
    avoided   = int(ext_tot * rate)
    bed_days  = avoided * extra
    cost_sav  = bed_days * cost_bed
    nurse_sav = bed_days * 0.5 * 1200
    readm_sav = avoided * 0.08 * 15000
    total     = cost_sav + nurse_sav + readm_sav
    annual    = total / years
    add_pts   = int(bed_days / avg_los_all) if avg_los_all > 0 else 0
    ind_rev   = add_pts * rev_pt
    ann_rev   = ind_rev / years
    ann_bed   = bed_days / years
    return dict(avoided=avoided, bed_days=bed_days, ann_bed=ann_bed,
                cost_sav=cost_sav, nurse_sav=nurse_sav, readm_sav=readm_sav,
                total=total, annual=annual, add_pts=add_pts,
                ind_rev=ind_rev, ann_rev=ann_rev)

# =========================================================
# PREDICTION
# =========================================================
def predict_risk(user_values: dict) -> float:
    if IS_PIPELINE:
        row = {}
        for col in feature_cols:
            if col in NUMERIC_COLS:
                row[col] = float(user_values.get(col, 0.0))
            else:
                row[col] = str(user_values.get(col, "Other/Unknown"))
        df_input = pd.DataFrame([row], columns=feature_cols)
        for col in NUMERIC_COLS:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)
        for col in CATEG_COLS:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
        return float(model.predict_proba(df_input)[0, 1])
    else:
        raw_cols = [c for c in feature_cols if c in dashboard.columns]
        ref      = dashboard[raw_cols].copy()
        new_row  = {}
        for col in raw_cols:
            if col in NUMERIC_COLS:
                new_row[col] = float(user_values.get(col, ref[col].median()))
            else:
                new_row[col] = str(user_values.get(col, "Other/Unknown"))
        new_df   = pd.DataFrame([new_row])
        combined = pd.concat([ref, new_df], ignore_index=True)
        cat_cols = [c for c in CATEG_COLS if c in combined.columns]
        encoded  = pd.get_dummies(combined, columns=cat_cols)
        expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(encoded.columns)
        new_enc  = encoded.iloc[[-1]].copy()
        for c in expected:
            if c not in new_enc.columns:
                new_enc[c] = 0.0
        new_enc = new_enc[expected].fillna(0.0)
        return float(model.predict_proba(new_enc)[0, 1])

# =========================================================
# HEADER — corrected F2 and Recall from notebook
# =========================================================
st.markdown(f"""
<div class="hdr">
  <div style="flex-shrink:0;">
    <svg width="90" height="90" viewBox="0 0 90 90" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="45" cy="45" r="42" fill="white" fill-opacity="0.10" stroke="white" stroke-opacity="0.25" stroke-width="1.5"/>
      <rect x="18" y="38" width="54" height="36" rx="4" fill="white" fill-opacity="0.15" stroke="white" stroke-opacity="0.4" stroke-width="1.5"/>
      <path d="M14 40 L45 16 L76 40" stroke="white" stroke-opacity="0.5" stroke-width="2" fill="white" fill-opacity="0.12"/>
      <circle cx="45" cy="46" r="15" fill="{COLORS['red']}" fill-opacity="0.90"/>
      <rect x="42" y="37" width="6" height="18" rx="2.5" fill="white"/>
      <rect x="35" y="44" width="20" height="6" rx="2.5" fill="white"/>
      <rect x="36" y="60" width="14" height="14" rx="2" fill="white" fill-opacity="0.30" stroke="white" stroke-opacity="0.35" stroke-width="1"/>
      <rect x="20" y="44" width="10" height="9" rx="2" fill="white" fill-opacity="0.40"/>
      <rect x="60" y="44" width="10" height="9" rx="2" fill="white" fill-opacity="0.40"/>
      <rect x="20" y="56" width="10" height="9" rx="2" fill="white" fill-opacity="0.30"/>
      <rect x="60" y="56" width="10" height="9" rx="2" fill="white" fill-opacity="0.30"/>
    </svg>
  </div>
  <div class="hdr-text">
    <div class="hdr-title">Hospital Stay Risk Intelligence</div>
    <div class="hdr-sub">Early prediction of extended hospital stays to support bed planning, staffing, resource allocation, and patient-flow decisions.</div>
    <div class="hdr-badges">
      <span class="badge">🤖 Logistic Regression</span>
      <span class="badge">⚖️ SMOTE Balanced</span>
      <span class="badge">🎯 Threshold: {THRESHOLD:.2f} (F2-Optimised)</span>
      <span class="badge">📈 Recall: {RECALL:.1%} </span>
      <span class="badge">🏥 Diagnosis-Relative Target (Median LOS)</span>
    </div>
    <div><span class="badge-src">📂 Source: Diabetes 130-US Hospitals (1999-2008) · UCI Machine Learning Repository</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺  Patient Triage",
    "📊  Executive Dashboard",
    "💰  Business Impact",
    "🔎  Patient Search",
])

FIG_W = 7; FIG_H = 4.2; FIG_H_TALL = 5.4

# =========================================================
# TAB 1 - PATIENT TRIAGE
# =========================================================
with tab1:
    st.markdown("#### 🩺 Clinical Triage Decision Support")
    st.markdown(
        "Enter key admission and early clinical indicators to estimate "
        "the patient's risk of an extended hospital stay."
    )
    st.markdown("---")

    left, right = st.columns([1, 1], gap="large")

    with left:
        section_hdr("Patient and Admission Inputs")
        age_midpoint = st.slider("Patient Age", 5, 95, 65, step=10)
        adm_type     = st.selectbox("Admission Type",
                                    sorted(dashboard["admission_type_label"].dropna().unique()))
        diag_grp     = st.selectbox("Primary Diagnosis Group", sorted(diag_map.keys()))
        diag_median  = float(diag_map.get(diag_grp, 4.0))

        st.markdown(
            f'<div class="inbox-steel">'
            f'<b>Diagnosis Group:</b> {diag_grp}<br>'
            f'<b>Median LOS for this group:</b> {diag_median:.2f} Days<br>'
            f'<span style="font-size:12px;color:{COLORS["slate"]};">'
            f'Patient is flagged extended if predicted stay exceeds this median LOS.'
            f'</span></div>',
            unsafe_allow_html=True
        )

        st.markdown(" ")
        section_hdr("Clinical Indicators")
        num_lab  = st.slider("Lab Procedures",       0, 120, 40)
        num_meds = st.slider("Medications",           1,  80, 15)
        num_proc = st.slider("Procedures",            0,   6,  1)
        num_diag = st.slider("Number of Diagnoses",   1,  16,  7)

        st.markdown(" ")
        section_hdr("Medication Status")
        diabetesMed = st.selectbox("Diabetes Medication",
                                   sorted(dashboard["diabetesMed"].dropna().unique()))
        insulin     = st.selectbox("Insulin Status",
                                   sorted(dashboard["insulin"].dropna().unique()))
        change      = st.selectbox("Medication Change", ["No","Ch"],
                                   format_func=lambda x: "Medication Changed" if x=="Ch" else "No Change")

        st.markdown(" ")
        section_hdr("Prior Utilisation")
        n_out = st.slider("Prior Outpatient Visits", 0, 40, 0)
        n_er  = st.slider("Prior Emergency Visits",  0, 40, 0)
        n_inp = st.slider("Prior Inpatient Visits",  0, 40, 0)

        prior_util = n_out + n_er + n_inp
        csi        = num_lab + num_proc + num_meds

        st.markdown(
            f'<div class="inbox-navy">'
            f'<b>Clinical Service Intensity:</b> {csi} '
            f'<span style="font-size:12px;">(Labs {num_lab} + Procedures {num_proc} + Medications {num_meds})</span><br>'
            f'<b>Prior Utilisation:</b> {prior_util} visits'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(" ")
        predict_btn = st.button("Generate Risk Assessment",
                                use_container_width=True, type="primary")

    with right:
        section_hdr("Risk Assessment Result")

        if predict_btn:
            age_bracket_map = {
                5:"[0-10)",15:"[10-20)",25:"[20-30)",35:"[30-40)",
                45:"[40-50)",55:"[50-60)",65:"[60-70)",
                75:"[70-80)",85:"[80-90)",95:"[90-100)"
            }
            adm_src_map = {
                "Emergency":"Emergency Room","Urgent":"Referral",
                "Elective":"Referral","Trauma Center":"Transfer/Other",
                "Other/Unknown":"Other/Unknown","Newborn":"Other/Unknown",
            }

            user_values = {
                "race":"Other/Unknown","gender":"Other/Unknown",
                "age":age_bracket_map.get(age_midpoint,"[60-70)"),
                "max_glu_serum":"Other/Unknown","A1Cresult":"Other/Unknown",
                "metformin":"No","repaglinide":"No","nateglinide":"No",
                "chlorpropamide":"No","glimepiride":"No","acetohexamide":"No",
                "glipizide":"No","glyburide":"No","tolbutamide":"No",
                "pioglitazone":"No","rosiglitazone":"No","acarbose":"No",
                "miglitol":"No","troglitazone":"No","tolazamide":"No",
                "examide":"No","citoglipton":"No",
                "insulin":str(insulin),
                "glyburide-metformin":"No","glipizide-metformin":"No",
                "glimepiride-pioglitazone":"No","metformin-rosiglitazone":"No",
                "metformin-pioglitazone":"No",
                "change":str(change),"diabetesMed":str(diabetesMed),
                "diag_1_group":str(diag_grp),
                "diag_2_group":"Other/Unknown","diag_3_group":"Other/Unknown",
                "admission_type_label":str(adm_type),
                "admission_source_group":adm_src_map.get(adm_type,"Other/Unknown"),
                "age_midpoint":           float(age_midpoint),
                "num_medications":        float(num_meds),
                "number_diagnoses":       float(num_diag),
                "num_lab_procedures":     float(num_lab),
                "num_procedures":         float(num_proc),
                "number_outpatient":      float(n_out),
                "number_emergency":       float(n_er),
                "number_inpatient":       float(n_inp),
                "prior_utilization":      float(prior_util),
                "clinical_service_intensity": float(csi),
                "medication_change_count":float(1 if change=="Ch" else 0),
                "diag_group_avg_los":     float(diag_median),  # field name kept for compat
                "prior_encounter_count":  0.0,
                "meds_per_diagnosis":     float(num_meds/(num_diag+1)),
                "labs_per_diagnosis":     float(num_lab/(num_diag+1)),
            }

            try:
                prob   = predict_risk(user_values)
                risk   = assign_risk(prob)

                card_cls = {"High Risk":"rcard-high","Medium Risk":"rcard-med","Low Risk":"rcard-low"}[risk]
                icon     = {"High Risk":"🔴","Medium Risk":"🟡","Low Risk":"🟢"}[risk]

                st.markdown(f'<div class="{card_cls}">{icon} {risk}</div>',
                            unsafe_allow_html=True)
                st.markdown(" ")

                m1, m2, m3 = st.columns(3)
                with m1: kpi("Probability Score",      f"{prob:.1%}",           top_color=COLORS["steel"])
                with m2: kpi("F2 Threshold",           f"{THRESHOLD:.2f}",      top_color=COLORS["navy"])
                with m3: kpi("Diag Group Median LOS",  f"{diag_median:.2f} Days", top_color=COLORS["mid"])
                st.markdown(" ")

                # ── Predicted LOS box ──────────────────────────────
                # Estimated LOS = median LOS of patients with similar risk score
                # Approximation: median_los * (1 + prob) gives a scaled estimate
                pred_los = diag_median * (1 + prob)
                los_color = COLORS["red"] if risk=="High Risk" else COLORS["amber"] if risk=="Medium Risk" else COLORS["green"]
                st.markdown(
                    f'<div class="ibox" style="border-left:6px solid {los_color}; text-align:center; padding:14px 20px; margin-bottom:10px;">'
                    f'<div style="font-size:11px; font-weight:700; color:{COLORS["slate"]}; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Predicted LOS Estimate</div>'
                    f'<div style="font-size:28px; font-weight:900; color:{los_color};">{pred_los:.1f} Days</div>'
                    f'<div style="font-size:11px; color:{COLORS["slate"]}; margin-top:3px;">Median LOS benchmark: {diag_median:.2f} Days · Diagnosis: {diag_grp}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── Gauge bar — improved readability ───────────────
                gauge_c = {"High Risk":COLORS["red"],"Medium Risk":COLORS["amber"],"Low Risk":COLORS["green"]}[risk]
                fig, ax = plt.subplots(figsize=(7, 1.4))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")

                # Background full bar
                ax.barh([""], [1.0], color="#E8EFF7", edgecolor="none", height=0.6)
                # Probability fill
                ax.barh([""], [prob], color=gauge_c, edgecolor="none", height=0.6)

                # Threshold lines
                ax.axvline(THRESHOLD, color=COLORS["navy"],
                           linestyle="--", linewidth=2.5, zorder=5)
                ax.axvline(THRESHOLD+0.15, color=COLORS["slate"],
                           linestyle=":", linewidth=2, zorder=5)

                # Score label — placed OUTSIDE bar on right side for readability
                label_x = min(prob + 0.03, 0.95)
                ax.text(label_x, 0,
                        f"  {prob:.1%}",
                        va="center", ha="left",
                        color=COLORS["navy"],
                        fontweight="bold", fontsize=14,
                        zorder=6)

                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability of Extended Stay", fontsize=12, color=COLORS["navy"])
                ax.set_yticks([])
                for sp in ["left","top","right"]:
                    ax.spines[sp].set_visible(False)

                # Legend positioned below chart
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0],[0], color=COLORS["navy"],  linestyle="--", linewidth=2.5,
                           label=f"Threshold {THRESHOLD:.2f}"),
                    Line2D([0],[0], color=COLORS["slate"], linestyle=":",  linewidth=2,
                           label=f"High Risk boundary {THRESHOLD+0.15:.2f}"),
                ]
                ax.legend(handles=legend_elements, loc="upper center",
                          bbox_to_anchor=(0.5, -0.65),
                          ncol=2, fontsize=10, frameon=True,
                          framealpha=0.9, edgecolor=COLORS["border"])
                plt.subplots_adjust(bottom=0.25)
                plt.show()
                st.pyplot(fig); plt.close(fig)


                st.markdown(" ")
                ac, abg = {"High Risk":(COLORS["red"],COLORS["redbg"]),"Medium Risk":(COLORS["amber"],COLORS["amberbg"]),"Low Risk":(COLORS["green"],COLORS["greenbg"])}[risk]
                st.markdown(
                    f'<div class="ibox" style="border-left:6px solid {ac}; background:{abg};">'
                    f'<b style="color:{ac};">Recommended Action</b><br>'
                    f'<span style="color:{COLORS["navy"]};">{action_text(risk)}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(" ")
                st.markdown(
                    f'<div class="inbox-navy">'
                    f'<b>Clinical Inputs Summary</b><br>'
                    f'Diagnosis: <b>{diag_grp}</b> · Median LOS Benchmark: <b>{diag_median:.2f} Days</b><br>'
                    f'Labs: <b>{num_lab}</b> · Medications: <b>{num_meds}</b> · Procedures: <b>{num_proc}</b> · Diagnoses: <b>{num_diag}</b><br>'
                    f'Service Intensity: <b>{csi}</b> · Medication Change: <b>{"Yes" if change=="Ch" else "No"}</b> · Insulin: <b>{insulin}</b><br>'
                    f'Prior Utilisation: <b>{prior_util}</b> (OP:{n_out} ER:{n_er} IP:{n_inp})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="warn-box">This score is a <b>clinical decision-support indicator</b>, not a diagnosis. Use alongside clinical judgement and care-team review.</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.warning(
                    "The saved model is a raw LogisticRegression (not a Pipeline). "
                    "Re-export from Colab using the pipeline export code and replace diabetes_los_model.pkl."
                )
        else:
            st.markdown(
                f'<div class="ready-box">'
                f'<div style="font-size:52px;margin-bottom:16px;">🏥</div>'
                f'<div style="font-size:18px;font-weight:800;color:{COLORS["navy"]};margin-bottom:10px;">Ready for Risk Assessment</div>'
                f'<div style="font-size:14px;color:{COLORS["slate"]};max-width:340px;margin:auto;line-height:1.6;">'
                f'Enter patient details on the left and click Generate Risk Assessment.'
                f'</div></div>',
                unsafe_allow_html=True
            )

# =========================================================
# TAB 2 - EXECUTIVE DASHBOARD
# =========================================================
with tab2:
    st.markdown("#### 📊 Executive Operational Dashboard")
    st.markdown(
        "Extended-stay risk patterns across all encounters for hospital leadership and operations teams."
    )
    st.markdown(
        f'<div style="font-size:11.5px;color:{COLORS["slate"]};background:{COLORS["pale"]};'
        f'padding:8px 14px;border-radius:8px;margin-bottom:12px;">'
        f'<b>Extended Stay Rate</b> = % of patients whose actual LOS exceeded their primary '
        f'diagnosis-group <b>median</b> LOS &nbsp;|&nbsp; '
        f'<b>Avg Risk Score</b> = mean of the model\'s predicted probability of extended stay '
        f'(0-1 scale) across all patients in the filtered cohort. '
        f'Example: 0.475 means the model gives an average 47.5% probability of extended stay.'
        f'</div>',
        unsafe_allow_html=True
    )

    section_hdr("Filter Records")
    f1, f2, f3, f4 = st.columns(4)
    with f1: rf  = st.selectbox("Risk Level",       ["All"]+risk_order, key="d_r")
    with f2: af  = st.selectbox("Admission Type",   ["All"]+sorted(dashboard["admission_type_label"].dropna().unique().tolist()), key="d_a")
    with f3: df_ = st.selectbox("Diagnosis Group",  ["All"]+sorted(dashboard["diag_1_group"].dropna().unique().tolist()), key="d_d")
    with f4: sf_ = st.selectbox("Admission Source", ["All"]+sorted(dashboard["admission_source_group"].dropna().unique().tolist()), key="d_s")

    dff = dashboard.copy()
    if rf  != "All": dff = dff[dff["Risk_Level"]            == rf]
    if af  != "All": dff = dff[dff["admission_type_label"]   == af]
    if df_ != "All": dff = dff[dff["diag_1_group"]           == df_]
    if sf_ != "All": dff = dff[dff["admission_source_group"] == sf_]

    n        = len(dff)
    ext_r    = dff["long_stay_target"].mean()*100 if n > 0 else 0.0
    # Median LOS — use median not mean
    med_los  = dff["time_in_hospital"].median()   if n > 0 else 0.0
    high_n   = int((dff["Risk_Level"]=="High Risk").sum()) if n > 0 else 0
    avg_sc   = dff["Probability_Score"].mean()    if n > 0 else 0.0

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Total Encounters",   f"{n:,}",             top_color=COLORS["slate"])
    with k2: kpi("Extended Stay Rate", f"{ext_r:.2f}%",      top_color=COLORS["navy"],
                 sub="% actual LOS > diag median")
    with k3: kpi("Median LOS",         f"{med_los:.2f} Days",top_color=COLORS["steel"])
    with k4: kpi("High Risk Cases",    f"{high_n:,}",        top_color=COLORS["red"])
    with k5: kpi("Avg Risk Score",     f"{avg_sc:.3f}",      top_color=COLORS["mid"],
                 sub="mean predicted prob (0-1)")
    st.markdown("---")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        chart_title("Encounters by Risk Level")
        rc = dff["Risk_Level"].value_counts().reindex(risk_order).fillna(0)
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor("white")
        ax.bar(rc.index, rc.values, color=risk_colors, edgecolor="white", linewidth=1.5, width=0.55)
        label_v(ax, fmt="{:,.0f}")
        ax.set_ylabel("Count", color=COLORS["slate"])
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        # Y axis as % not decimal
        chart_title("Actual Extended Stay Rate by Risk Level (%)")
        rr = dff.groupby("Risk_Level")["long_stay_target"].mean().reindex(risk_order).fillna(0) * 100
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor("white")
        bars = ax.bar(rr.index, rr.values, color=risk_colors, edgecolor="white", linewidth=1.5, width=0.55)
        for bar, val in zip(bars, rr.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{val:.1f}%", ha="center",
                    fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Extended Stay Rate (%)", color=COLORS["slate"])
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(" ")

    # Row 2 — replaced static Key Drivers with dynamic Actual vs Median LOS by Diagnosis Group
    c3, c4 = st.columns(2)
    with c3:
        chart_title("Extended Stay Rate by Admission Type (Dark = Highest)")
        ar = dff.groupby("admission_type_label")["long_stay_target"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
        fig.patch.set_facecolor("white")
        bc = bar3_colors(ar.values)
        ax.barh(ar.index, ar.values, color=bc, edgecolor="white", linewidth=1.2, height=0.55)
        for bar, val in zip(ax.patches, ar.values):
            ax.text(bar.get_width() + ar.max()*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center",
                    fontsize=11, fontweight="bold", color=COLORS["navy"])
        ax.set_xlabel("Extended Stay Rate (%)", color=COLORS["slate"])
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c4:
        # Dynamic: Actual Median LOS by Diagnosis Group — reacts to filters
        chart_title("Actual Median LOS by Diagnosis Group (Days)")
        med_by_diag = (dff.groupby("diag_1_group")["time_in_hospital"]
                       .median().sort_values(ascending=True))
        bc = bar3_colors(med_by_diag.values)
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
        fig.patch.set_facecolor("white")
        ax.barh(med_by_diag.index, med_by_diag.values,
                color=bc, edgecolor="white", linewidth=1.2, height=0.55)
        for bar, val in zip(ax.patches, med_by_diag.values):
            ax.text(bar.get_width() + med_by_diag.max()*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}d", va="center",
                    fontsize=11, fontweight="bold", color=COLORS["navy"])
        ax.set_xlabel("Median LOS (Days)", color=COLORS["slate"])
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(" ")

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        chart_title("Extended Stay Rate by Diagnosis Group (%)")
        dr = (dff.groupby("diag_1_group")["long_stay_target"]
              .mean().sort_values(ascending=True).head(10)) * 100
        bc = bar3_colors(dr.values)
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
        fig.patch.set_facecolor("white")
        ax.barh(dr.index, dr.values, color=bc, edgecolor="white", linewidth=1.2, height=0.55)
        for bar, val in zip(ax.patches, dr.values):
            ax.text(bar.get_width()+dr.max()*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center",
                    fontsize=11, fontweight="bold", color=COLORS["navy"])
        ax.set_xlabel("Extended Stay Rate (%)", color=COLORS["slate"])
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c6:
        chart_title("Resource Utilisation by Risk Level")
        res_c = [c for c in ["num_medications","number_diagnoses","clinical_service_intensity"]
                 if c in dff.columns]
        if res_c and n > 0:
            res = dff.groupby("Risk_Level")[res_c].mean().reindex(risk_order).reset_index()
            rm  = res.melt(id_vars="Risk_Level", var_name="Resource", value_name="Avg")
            rm["Resource"] = rm["Resource"].map({
                "num_medications":            "Medications",
                "number_diagnoses":           "Diagnoses",
                "clinical_service_intensity": "Service Intensity"
            }).fillna(rm["Resource"])
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
            fig.patch.set_facecolor("white")
            sns.barplot(data=rm, x="Resource", y="Avg",
                        hue="Risk_Level", palette=risk_colors,
                        edgecolor="white", linewidth=1.2, ax=ax)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", padding=2,
                             fontsize=11, fontweight="bold", color=COLORS["navy"])
            ax.set_ylabel("Average per Patient", color=COLORS["slate"])
            ax.legend(title="Risk Level", fontsize=11,
                      framealpha=0.9, edgecolor=COLORS["border"])
            clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Triage Summary Table
    st.markdown("---")
    st.markdown("### Triage Performance Summary")
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["slate"]};margin-bottom:8px;">'
        f'<b>Extended Stay Rate</b> = % of patients in each risk band whose actual LOS exceeded their '
        f'diagnosis-group <b>median</b> LOS. '
        f'<b>Avg Probability</b> = mean model-predicted probability score (0-1) for that band.'
        f'</div>',
        unsafe_allow_html=True
    )
    td = triage_df.copy()
    for col in td.select_dtypes(include=[np.number]).columns:
        td[col] = td[col].round(2)
    td = td.rename(columns={
        "Risk_Level":                "Risk Level",
        "patient_count":             "Patient Count",
        "avg_probability":           "Avg Probability (0-1)",
        "actual_extended_stay_rate": "Actual Extended Stay Rate",
        "median_los":                "Median Actual LOS (Days)",
        "avg_medications":           "Avg Medications",
        "avg_diagnoses":             "Avg Diagnoses",
    })
    st.dataframe(td, use_container_width=True)

# =========================================================
# TAB 3 - BUSINESS IMPACT
# =========================================================
with tab3:
    st.markdown("#### 💰 Business Impact Analysis")
    st.markdown("Financial and operational impact of reducing extended hospital stays through early AI-driven risk identification.")

    ext_tot     = int(dashboard["long_stay_target"].sum())
    # Use median LOS for each group
    short_med   = dashboard.loc[dashboard["long_stay_target"]==0,"time_in_hospital"].median()
    long_med    = dashboard.loc[dashboard["long_stay_target"]==1,"time_in_hospital"].median()
    extra       = long_med - short_med
    avg_los_all = dashboard["time_in_hospital"].median()   # median for patient throughput calc

    st.markdown("---")
    section_hdr("Adjustable Assumptions — All KPIs and Charts Update Automatically")

    a1, a2, a3 = st.columns(3)
    with a1: cost_bed = float(st.number_input("Cost per Bed-Day ($)",              value=2500, step=100, min_value=100, key="bi_cost"))
    with a2: rev_pt   = float(st.number_input("Revenue per Additional Patient ($)", value=8000, step=500, min_value=100, key="bi_rev"))
    with a3: years    = float(st.number_input("Dataset Years (for annualising)",    value=10,   step=1,   min_value=1,   key="bi_years"))

    s5  = calc_scenario(ext_tot, extra, avg_los_all, 0.05, cost_bed, rev_pt, years)
    s10 = calc_scenario(ext_tot, extra, avg_los_all, 0.10, cost_bed, rev_pt, years)
    s15 = calc_scenario(ext_tot, extra, avg_los_all, 0.15, cost_bed, rev_pt, years)
    scen_list   = [s5, s10, s15]
    scen_labels = ["5% Reduction","10% Reduction","15% Reduction"]

    st.markdown("---")
    b1,b2,b3,b4 = st.columns(4)
    with b1: kpi("Total Extended Cases",   f"{ext_tot:,}",          top_color=COLORS["navy"])
    with b2: kpi("Median LOS Short Stay",  f"{short_med:.2f} Days", top_color=COLORS["slate"])
    with b3: kpi("Median LOS Extended",    f"{long_med:.2f} Days",  top_color=COLORS["steel"])
    with b4: kpi("Extra Median Days/Case", f"{extra:.2f} Days",     top_color=COLORS["mid"])
    st.markdown("---")

    st.markdown(
        f'<div style="font-size:11.5px;color:{COLORS["slate"]};background:{COLORS["pale"]};padding:8px 14px;border-radius:8px;margin-bottom:12px;">'
        f'<b>Total Impact</b> = cumulative savings across the full {int(years)}-year period &nbsp;|&nbsp; '
        f'<b>Annual Impact</b> = Total divided by {int(years)} years &nbsp;|&nbsp; '
        f'<b>Annual Bed-Days</b> = Total Bed-Days divided by {int(years)} years'
        f'</div>',
        unsafe_allow_html=True
    )

    sp1,sp2,sp3,sp4 = st.columns(4)
    with sp1: kpi("Total Impact (10%)",              f"${s10['total']:,.0f}",   top_color=COLORS["navy"])
    with sp2: kpi("Annual Impact (10%)",             f"${s10['annual']:,.0f}",  top_color=COLORS["steel"])
    with sp3: kpi("Annual Bed-Days Recovered (10%)", f"{s10['ann_bed']:,.0f}",  top_color=COLORS["slate"])
    with sp4: kpi("Annual Indirect Revenue (10%)",   f"${s10['ann_rev']:,.0f}", top_color=COLORS["mid"])
    st.markdown("---")

    ch1, ch2 = st.columns(2)
    with ch1:
        chart_title("Total Financial Impact by Scenario")
        totals = [s["total"] for s in scen_list]
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        bars = ax.bar(scen_labels, totals, color=bar3_colors(totals), edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, totals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(totals)*0.01,
                    f"${val/1e6:.2f}M", ha="center", fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Total Impact ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        chart_title("Annual Impact by Scenario")
        annuals = [s["annual"] for s in scen_list]
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        bars = ax.bar(scen_labels, annuals, color=bar3_colors(annuals), edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, annuals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(annuals)*0.01,
                    f"${val/1e6:.2f}M", ha="center", fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Annual Impact ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(" ")
    ch3, ch4 = st.columns(2)
    with ch3:
        chart_title("Savings Breakdown by Component")
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        x = np.arange(3); w = 0.5
        cv = [s["cost_sav"]  for s in scen_list]
        nv = [s["nurse_sav"] for s in scen_list]
        rv = [s["readm_sav"] for s in scen_list]
        ax.bar(x, cv, w, color=COLORS["navy"],      edgecolor="white", label="Bed-Day Savings")
        ax.bar(x, nv, w, bottom=cv,                  color=COLORS["steel"],     edgecolor="white", label="Nurse Savings")
        ax.bar(x, rv, w, bottom=[c+n for c,n in zip(cv,nv)],
               color=COLORS["lightblue"], edgecolor="white", label="Readmission Savings")
        ax.set_xticks(x); ax.set_xticklabels(scen_labels)
        ax.set_ylabel("Savings ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
        ax.legend(fontsize=11, framealpha=0.9, edgecolor=COLORS["border"])
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch4:
        chart_title("Annual Bed-Days Recovered by Scenario")
        ann_beds = [s["ann_bed"] for s in scen_list]
        fig, ax  = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        bars = ax.bar(scen_labels, ann_beds, color=bar3_colors(ann_beds), edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, ann_beds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(ann_beds)*0.01,
                    f"{val:,.0f}", ha="center", fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Annual Bed-Days")
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown(" ")
    ch5, ch6 = st.columns(2)
    with ch5:
        chart_title("Additional Patients Treatable (Total Dataset Period)")
        pt_vals = [s["add_pts"] for s in scen_list]
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        bars = ax.bar(scen_labels, pt_vals, color=bar3_colors(pt_vals), edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, pt_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(pt_vals)*0.01,
                    f"{int(val):,}", ha="center", fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Patients")
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch6:
        chart_title("Annual Indirect Revenue by Scenario")
        ann_revs = [s["ann_rev"] for s in scen_list]
        fig, ax  = plt.subplots(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        bars = ax.bar(scen_labels, ann_revs, color=bar3_colors(ann_revs), edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, ann_revs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(ann_revs)*0.01,
                    f"${val/1e6:.2f}M", ha="center", fontweight="bold", fontsize=12, color=COLORS["navy"])
        ax.set_ylabel("Annual Revenue ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
        clean(ax); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.markdown("### Full Scenario Table")
    disp_data = []
    for label, s in zip(scen_labels, scen_list):
        disp_data.append({
            "Scenario":                      label,
            "Avoided Cases":                 f"{s['avoided']:,}",
            "Total Bed-Days Recovered":      f"{s['bed_days']:,.2f}",
            "Annual Bed-Days Recovered":     f"{s['ann_bed']:,.2f}",
            "Cost Savings ($)":              f"${s['cost_sav']:,.2f}",
            "Nurse Savings ($)":             f"${s['nurse_sav']:,.2f}",
            "Readmission Savings ($)":       f"${s['readm_sav']:,.2f}",
            "Total Financial Impact ($)":    f"${s['total']:,.2f}",
            "Annual Financial Impact ($)":   f"${s['annual']:,.2f}",
            "Additional Patients Treatable": f"{s['add_pts']:,}",
            "Total Indirect Revenue ($)":    f"${s['ind_rev']:,.2f}",
            "Annual Indirect Revenue ($)":   f"${s['ann_rev']:,.2f}",
        })
    st.dataframe(pd.DataFrame(disp_data), use_container_width=True)

    st.markdown("---")
    st.markdown(
        f'<div class="ibox">'
        f'<b style="color:{COLORS["navy"]};font-size:15px;">Business Impact Summary - 10% Reduction Scenario</b><br><br>'
        f'A <b>10% reduction</b> in extended-stay cases avoids <b>{s10["avoided"]:,} unnecessary extended admissions</b>, '
        f'recovering <b>{s10["bed_days"]:,.2f} bed-days</b> total across the {int(years)}-year dataset period '
        f'(<b>{s10["ann_bed"]:,.2f} bed-days per year</b>).<br><br>'
        f'This generates an estimated <b>${s10["total"]:,.2f}</b> in total financial impact, '
        f'equivalent to <b>${s10["annual"]:,.2f} per year</b>. '
        f'Freed capacity enables treatment of approximately <b>{s10["add_pts"]:,} additional patients</b>, '
        f'generating <b>${s10["ann_rev"]:,.2f} in annual indirect revenue</b>.'
        f'</div>',
        unsafe_allow_html=True
    )

# =========================================================
# TAB 4 - PATIENT SEARCH
# =========================================================
with tab4:
    st.markdown("#### 🔎 Patient Search and Risk Review")
    st.markdown("Search and filter individual patient records by risk level, diagnosis group, admission type, and probability score.")
    st.markdown("---")
    section_hdr("Search and Filter")

    s1, s2, s3 = st.columns(3)
    with s1: s_risk = st.multiselect("Risk Level",      risk_order, default=risk_order, key="sr")
    with s2: s_diag = st.multiselect("Diagnosis Group", sorted(dashboard["diag_1_group"].dropna().unique()), default=[], key="sd")
    with s3: s_adm  = st.multiselect("Admission Type",  sorted(dashboard["admission_type_label"].dropna().unique()), default=[], key="sa")

    s4, s5_ = st.columns([2,1])
    with s4:  s_prob = st.slider("Minimum Probability Score", 0.0, 1.0, 0.0, 0.01, key="sp")
    with s5_: s_enc  = st.text_input("Search Encounter ID", key="se")

    sf2 = dashboard.copy()
    if s_risk: sf2 = sf2[sf2["Risk_Level"].isin(s_risk)]
    if s_diag: sf2 = sf2[sf2["diag_1_group"].isin(s_diag)]
    if s_adm:  sf2 = sf2[sf2["admission_type_label"].isin(s_adm)]
    sf2 = sf2[sf2["Probability_Score"] >= s_prob]
    if s_enc:
        sf2 = sf2[sf2["encounter_id"].astype(str).str.contains(s_enc, case=False, na=False)]

    sn = len(sf2)
    st.markdown("---")

    sk1,sk2,sk3,sk4 = st.columns(4)
    with sk1: kpi("Matching Patients",    f"{sn:,}",   top_color=COLORS["slate"])
    with sk2: kpi("High Risk",            f"{int((sf2['Risk_Level']=='High Risk').sum()):,}", top_color=COLORS["red"])
    with sk3: kpi("Avg Probability Score",
                  f"{sf2['Probability_Score'].mean():.3f}" if sn>0 else "0.000",
                  sub="mean predicted prob (0-1)", top_color=COLORS["steel"])
    with sk4: kpi("Median LOS",
                  f"{sf2['time_in_hospital'].median():.2f} Days" if sn>0 else "0.00 Days",
                  top_color=COLORS["mid"])
    st.markdown("---")

    show_cols = [c for c in [
        "encounter_id","age","gender","admission_type_label","admission_source_group",
        "diag_1_group","num_medications","number_diagnoses","time_in_hospital",
        "long_stay_target","Probability_Score","Risk_Level","Suggested_Action"
    ] if c in sf2.columns]

    if sn > 0:
        st.markdown(f"**{sn:,} patients** match current filters, sorted by risk score (highest first).")
        st.download_button(
            "Download Patient List",
            sf2[show_cols].sort_values("Probability_Score", ascending=False)
            .to_csv(index=False).encode("utf-8"),
            "patient_search_results.csv", "text/csv"
        )

        display_df = (sf2[show_cols]
                      .sort_values("Probability_Score", ascending=False)
                      .reset_index(drop=True))

        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(2)

        display_df = display_df.rename(columns={
            "encounter_id":           "Encounter ID",
            "age":                    "Age Group",
            "gender":                 "Gender",
            "admission_type_label":   "Admission Type",
            "admission_source_group": "Admission Source",
            "diag_1_group":           "Primary Diagnosis Group",
            "num_medications":        "Medications",
            "number_diagnoses":       "No. of Diagnoses",
            "time_in_hospital":       "Actual LOS (Days)",
            "long_stay_target":       "Extended Stay (1=Yes)",
            "Probability_Score":      "Risk Score (0-1)",
            "Risk_Level":             "Risk Level",
            "Suggested_Action":       "Suggested Action",
        })

        def highlight_risk_renamed(row):
            styles = {
                "High Risk":   f"background-color:{COLORS['redbg']};color:{COLORS['red']};font-weight:700",
                "Medium Risk": f"background-color:{COLORS['amberbg']};color:{COLORS['amber']};font-weight:600",
                "Low Risk":    f"background-color:{COLORS['greenbg']};color:{COLORS['green']}",
            }
            return [styles.get(row.get("Risk Level",""), "")] * len(row)

        if len(display_df) <= 2000:
            st.dataframe(display_df.style.apply(highlight_risk_renamed, axis=1),
                         use_container_width=True, height=540)
        else:
            st.dataframe(display_df, use_container_width=True, height=540)
            
    else:
        st.warning("No records match the selected filters.")