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
    f_str = st.text_input("Funktion f(x) eingeben", "x**3")
    if f_str:
        try:
            df = sp.diff(sp.sympify(f_str), x)
            st.latex(r"\frac{d}{dx}f(x) = " + sp.latex(df))
        except Exception as e:
            st.error(e)

# 3. Gleichung lösen
elif auswahl == "🧮 Gleichung lösen":
    st.header("🧮 Gleichungslöser")
    eqn = st.text_input("Gleichung eingeben (z. B. x**2 - 4 = 0)", "x**2 - 4 = 0")
    if "=" in eqn:
        try:
            lhs, rhs = eqn.split("=")
            sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
            st.success(f"Lösungen: {sol}")
        except Exception as e:
            st.error(e)

# 4. DGL 2. Ordnung
elif auswahl == "📈 DGL 2. Ordnung":
    st.header("📈 DGL 2. Ordnung")
    ode = st.text_input("DGL eingeben (z. B. y'' + y)", "y'' + y")
    y = Function('y')
    try:
        eq = Eq(eval(ode.replace("y''", "y(t).diff(t,2)").replace("y'", "y(t).diff(t)")), 0)
        sol = dsolve(eq, y(t))
        st.latex("y(t) = " + sp.latex(sol.rhs))
    except Exception as e:
        st.error(e)

# 5. DGL 1. Ordnung
elif auswahl == "🧊 DGL 1. Ordnung":
    st.header("🧊 DGL 1. Ordnung")
    f_str = st.text_input("dy/dt = f(t,y)", "-2*y + 1")
    y0 = st.number_input("Anfangswert y(0)", value=1.0)
    T = st.slider("Zeitbereich [s]", 5, 50, 10)
    try:
        f = lambda t, y: eval(f_str, {"t": t, "y": y, "np": np})
        sol = solve_ivp(f, [0, T], [y0], t_eval=np.linspace(0, T, 500))
        plot_loesung(sol.t, sol.y[0])
    except Exception as e:
        st.error(e)

# 6. Regelungstechnik
elif auswahl == "⚙️ Regelungstechnik":
    st.header("⚙️ Regelungstechnik – Sprungantwort")
    m = st.slider("Masse m", 0.1, 10.0, 1.0)
    c = st.slider("Dämpfung c", 0.0, 10.0, 1.0)
    k = st.slider("Federkonstante k", 1.0, 100.0, 20.0)
    sys = tf([1], [m, c, k])
    t_vals = np.linspace(0, 10, 1000)
    t, y = step_response(sys, t_vals)
    plot_loesung(t, y)

# 7. Lagrange
elif auswahl == "🧲 Lagrange-Mechanik":
    st.header("🧲 Lagrange-Mechanik")
    q = Function('q')(t)
    m_sym, k_sym = sp.symbols('m k')
    L = (1/2)*m_sym*sp.diff(q, t)**2 - (1/2)*k_sym*q**2
    eq = sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)
    st.latex("L = T - V = " + sp.latex(L))
    st.latex(sp.latex(eq) + " = 0")

# 8. Balkenkräfte
elif auswahl == "📐 Balkenkräfte":
    st.header("📐 Reaktionskräfte eines Balkens")
    Lb = st.number_input("Balkenlänge L [m]", 1.0)
    F = st.number_input("Zentrische Last F [N]", 100.0)
    st.success(f"Reaktionskräfte: A = {F/2:.2f} N, B = {F/2:.2f} N")

# 9. Matrizen
elif auswahl == "📊 Matrizenwerkzeuge":
    st.header("📊 Matrix-Rechner")
    A = st.text_input("Matrix A (z. B. [[2,1],[1,3]])", "[[2,1],[1,3]]")
    b = st.text_input("Vektor b (z. B. [4,5])", "[4,5]")
    try:
        A_mat = sp.Matrix(eval(A))
        b_vec = sp.Matrix(eval(b))
        st.write("Lösung x =", A_mat.LUsolve(b_vec))
        st.write("Determinante det(A) =", A_mat.det())
        st.write("Eigenwerte =", A_mat.eigenvals())
    except Exception as e:
        st.error(e)

# 10. Laplace
elif auswahl == "🔄 Laplace-Transformation":
    st.header("🔄 Laplace-Transformation")
    f_str = st.text_input("f(t)", "exp(-t)*sin(t)")
    try:
        F = laplace_transform(sp.sympify(f_str), t, s, noconds=True)
        st.latex(r"\mathcal{L}\{" + sp.latex(sp.sympify(f_str)) + r"\} = " + sp.latex(F))
    except Exception as e:
        st.error(e)

# 11. RLC-Schaltung
elif auswahl == "⚡ RLC-Schaltung":
    st.header("⚡ Impedanz einer RLC-Schaltung")
    R = st.number_input("Widerstand R [Ω]", 1.0)
    L_val = st.number_input("Induktivität L [H]", 0.1)
    C_val = st.number_input("Kapazität C [F]", 0.01)
    freq = st.slider("Frequenz [Hz]", 10, 1000, 50)
    w = 2 * math.pi * freq
    Z = complex(R, w * L_val - 1 / (w * C_val))
    st.success(f"Z = {Z.real:.2f} + j{Z.imag:.2f} Ω")

# 12. Einheitenrechner
elif auswahl == "🧾 Einheitenrechner":
    st.header("🧾 Einheiten umrechnen")
    opt = st.selectbox("Umrechnung", ["N→kgf", "kgf→N", "Pa→bar", "bar→Pa", "m/s→km/h", "km/h→m/s"])
    val = st.number_input("Wert", value=1.0)
    conv = {
        "N→kgf": lambda x: x / 9.80665,
        "kgf→N": lambda x: x * 9.80665,
        "Pa→bar": lambda x: x / 1e5,
        "bar→Pa": lambda x: x * 1e5,
        "m/s→km/h": lambda x: x * 3.6,
        "km/h→m/s": lambda x: x / 3.6
    }
    st.success(f"Ergebnis: {conv[opt](val):.4f}")

# 13. Biegemomentdiagramm (neu)
elif auswahl == "📌 Biegemomentdiagramm":
    st.header("📌 Biegemoment-Diagramm für Einfeldträger")
    L = st.number_input("Trägerlänge L [m]", 5.0)
    P = st.number_input("Einzellast P [N] (mittig)", 100.0)
    x_vals = np.linspace(0, L, 500)
    M = np.where(x_vals < L/2, P * x_vals / 2, P * (L - x_vals) / 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=M, mode='lines', name='M(x)'))
    fig.update_layout(title='Biegemomentverlauf', xaxis_title='x [m]', yaxis_title='M(x) [Nm]')
    st.plotly_chart(fig)

# 14. Kinematik-Rechner (neu)
elif auswahl == "🚗 Kinematik-Rechner":
    st.header("🚗 Kinematik-Rechner (gleichmäßig beschleunigte Bewegung)")
    v0 = st.number_input("Anfangsgeschwindigkeit v₀ [m/s]", 0.0)
    a = st.number_input("Beschleunigung a [m/s²]", 1.0)
    t_end = st.slider("Zeitdauer t [s]", 1, 20, 5)
    t_vals = np.linspace(0, t_end, 200)
    s_vals = v0 * t_vals + 0.5 * a * t_vals**2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals, y=s_vals, mode='lines', name='s(t)'))
    fig.update_layout(title='Orts-Zeit-Verlauf', xaxis_title='t [s]', yaxis_title='s(t) [m]')
    st.plotly_chart(fig)
