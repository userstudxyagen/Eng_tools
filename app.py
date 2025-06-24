# 🇩🇪 Moderne und Ästhetische Version des Engineering-Helfer-Tools (auf Deutsch)

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from scipy.integrate import solve_ivp
from control import tf, step_response
from sympy.abc import s, t
from sympy import Function, Eq, dsolve, laplace_transform

st.set_page_config(page_title="Ingenieur-Helfer", layout="wide")
st.title("🔧 Ingenieur-Helfer (Deutsch)")

# Hilfsfunktion für Tool-Informationen
def tool_info(title, description, usage, examples=None):
    with st.expander("ℹ️ Info"):
        st.markdown(f"**{title}**")
        st.markdown(f"**Beschreibung:** {description}")
        st.markdown(f"**Verwendung:** {usage}")
        if examples:
            st.markdown("**Beispiele:**")
            for example in examples:
                st.markdown(f"- {example}")

# Sidebar mit Werkzeugauswahl
st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/67/000000/external-engineering-industrial-engineering-flat-icons-inmotus-design.png", width=80)
st.sidebar.title("🛠️ Wähle ein Werkzeug")

werkzeuge = [
    "🔢 Integrieren", "📉 Ableiten", "🧮 Gleichung lösen", "📈 DGL 2. Ordnung",
    "🧊 DGL 1. Ordnung", "⚙️ Regelungstechnik", "🧲 Lagrange-Mechanik",
    "📐 Balkenkräfte", "📊 Matrizenwerkzeuge", "🔄 Laplace-Transformation",
    "⚡ RLC-Schaltung", "🧾 Einheitenrechner",
    "📌 Biegemomentdiagramm", "🚗 Kinematik-Rechner"
]
auswahl = st.sidebar.radio("Werkzeug", werkzeuge)

x = sp.Symbol('x')

def plot_loesung(t, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y(t)'))
    fig.update_layout(title='Lösungsdiagramm', xaxis_title='t', yaxis_title='y(t)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# 1. Integral
if auswahl == "🔢 Integrieren":
    st.header("🔢 Integralrechner")
    tool_info(
        "Integralrechner",
        "Berechnet das unbestimmte oder bestimmte Integral einer Funktion.",
        "Geben Sie eine Funktion f(x) ein, die integriert werden soll.",
        ["x**2 → 1/3 x³ + C", "sin(x) → -cos(x) + C"]
    )
    f_str = st.text_input("Funktion f(x) eingeben", "x**2")
    if f_str:
        try:
            integral = sp.integrate(sp.sympify(f_str), x)
            st.latex(r"\int f(x) dx = " + sp.latex(integral) + "+ C")
        except Exception as e:
            st.error(e)

# 2. Ableitung
elif auswahl == "📉 Ableiten":
    st.header("📉 Ableitungsrechner")
    tool_info(
        "Ableitungsrechner",
        "Berechnet die Ableitung einer Funktion nach x.",
        "Geben Sie eine Funktion f(x) ein, die abgeleitet werden soll.",
        ["x³ → 3x²", "sin(x) → cos(x)", "e^x → e^x"]
    )
    f_str = st.text_input("Funktion f(x) eingeben", "x**3")
    if f_str:
        try:
            df = sp.diff(sp.sympify(f_str), x)
            st.latex(r"\frac{d}{dx}f(x) = " + sp.latex(df))
        except Exception as e:
            st.error(e)

# Rest des Codes bleibt gleich...
