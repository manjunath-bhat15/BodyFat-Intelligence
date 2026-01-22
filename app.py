import joblib
import json
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Required for stable execution on Hugging Face
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# 1. ASSET LOADING
# =========================
model_with_density = joblib.load("rf_with_density.pkl")
model_no_density = joblib.load("rf_no_density.pkl")

with open("feature_importance_v5.json") as f:
    FEATURE_IMPORTANCE = json.load(f)

FEATURE_ORDER = ["Density", "Age", "Weight", "Height", "Abdomen", "Neck", "Chest", "Hip", "Thigh"]
HISTORY = []

# =========================
# 2. LOGIC & STABILITY FIXES
# =========================
def get_health_status(val):
    if val < 8: return "ESSENTIAL FAT", "#38bdf8"
    if val < 14: return "ATHLETE / FIT", "#10b981"
    if val < 22: return "FITNESS / HEALTHY", "#22c55e"
    if val < 28: return "AVERAGE", "#f59e0b"
    return "OBESE RANGE", "#ef4444"

def predict(name, density, age, weight, height, abdomen, neck, chest, hip, thigh, engine):
    # CRITICAL: Close all open plots to prevent Internal Server Errors
    plt.close('all') 
    
    try:
        data = {"Density": density, "Age": age, "Weight": weight, "Height": height,
                "Abdomen": abdomen, "Neck": neck, "Chest": chest, "Hip": hip, "Thigh": thigh}

        # Model Routing
        is_density = engine == "With Density"
        model = model_with_density if is_density else model_no_density
        features = FEATURE_ORDER if is_density else [f for f in FEATURE_ORDER if f != "Density"]
        fi_key = "Accuracy-focused (with Density)" if is_density else "Explainability-focused (no Density)"

        # Core Prediction
        X = pd.DataFrame([[data[f] for f in features]], columns=features)
        pred = round(float(model.predict(X)[0]), 2)
        status, color = get_health_status(pred)

        # 🎯 Goal Logic: Target 15% Body Fat
        lean_mass = weight * (1 - (pred / 100))
        target_weight = round(lean_mass / (1 - 0.15), 1)
        w_diff = round(weight - target_weight, 1)

        # Generate Visuals
        fi = FEATURE_IMPORTANCE[fi_key]
        items = sorted(fi.items(), key=lambda x: x[1])
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        ax_fi.barh([k for k, _ in items], [v for _, v in items], color="#38bdf8")
        ax_fi.set_title("Impact of Your Metrics", color="white")
        plt.tight_layout()

        # Update History
        HISTORY.append({"time": datetime.now().strftime("%H:%M:%S"), "user": name or "Guest", "bf": pred})
        
        # Result Dashboard
        result_html = f"""
        <div style="background: #1e293b; padding: 25px; border-radius: 15px; border-top: 6px solid {color}; box-shadow: 0 10px 15px rgba(0,0,0,0.3);">
            <p style="margin: 0; color: #94a3b8; font-size: 0.8rem; letter-spacing: 2px;">NEURAL ANALYSIS RESULT</p>
            <h1 style="margin: 10px 0; color: {color}; font-size: 3.8rem; font-weight: 900;">{pred}%</h1>
            <h3 style="margin: 0; color: {color};">{status}</h3>
            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #334155; color: #cbd5e1; font-size: 0.9rem;">
                📈 <b>Goal Check:</b> To reach 15%, your target weight is <b>{target_weight}kg</b> (current diff: {w_diff}kg)
            </div>
        </div>
        """

        return result_html, fig_fi, pd.DataFrame(HISTORY[-5:][::-1])

    except Exception as e:
        return f"<div style='background:#450a0a; padding:15px; border-radius:10px; color:#fecaca;'><b>Error:</b> {str(e)}</div>", None, None

# =========================
# 3. PREMIUM UI (DASHBOARD)
# =========================
CSS = """
.gradio-container { background: #020617 !important; color: white !important; }
.tabs { border: none !important; }
.input-group { background: #0f172a; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 **BodyFat Intelligence**")
    
    with gr.Row():
        # Input Section
        with gr.Column(scale=4):
            with gr.Group(elem_classes="input-group"):
                name = gr.Textbox(label="User Identifier", placeholder="Enter name...")
                engine = gr.Radio(["With Density", "Without Density"], value="With Density", label="Prediction Engine")
            
            with gr.Tabs():
                with gr.TabItem("🚀 Quick Vitals"):
                    den = gr.Slider(0.9, 1.1, value=1.07, step=0.001, label="Body Density")
                    age = gr.Slider(18, 80, value=30, label="Age")
                    wgt = gr.Slider(40, 150, value=75, label="Weight (kg)")
                    hgt = gr.Slider(140, 200, value=175, label="Height (cm)")
                
                with gr.TabItem("📐 Detailed Body Map"):
                    with gr.Row():
                        abd = gr.Number(label="Abdomen", value=85)
                        nck = gr.Number(label="Neck", value=37)
                    with gr.Row():
                        chs = gr.Number(label="Chest", value=95)
                        hip = gr.Number(label="Hip", value=94)
                    thg = gr.Number(label="Thigh", value=55)
            
            run_btn = gr.Button("⚡ EXECUTE ANALYTICS", variant="primary")

        # Output Section
        with gr.Column(scale=5):
            res_html = gr.HTML("<div style='text-align:center; padding: 60px; color: #475569;'>System Standby... Enter data to begin.</div>")
            with gr.Accordion("📊 Feature Interpretability", open=True):
                plot_fi = gr.Plot()
            with gr.Accordion("🕒 Recent Session History", open=True):
                history_table = gr.Dataframe(interactive=False)

    run_btn.click(predict, [name, den, age, wgt, hgt, abd, nck, chs, hip, thg, engine], [res_html, plot_fi, history_table])

demo.launch()