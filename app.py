# Modern and Aesthetic Version of the Engineering Helper App

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

st.set_page_config(page_title="Engineering Helper (Modern)", layout="wide")
st.title("🚀 Engineering Helper Dashboard")
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 10px 16px;
        margin: 0 2px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/67/000000/external-engineering-industrial-engineering-flat-icons-inmotus-design.png", width=80)
st.sidebar.title("🔧 Select a Tool")

tools = [
    "🔢 Integrate", "📉 Derive", "🧮 Solve Eq", "📈 ODE 2nd", "🧊 ODE 1st",
    "⚙️ Control", "🧲 Lagrangian", "📐 Beam Forces", "📊 Matrix Tools",
    "🔄 Laplace", "⚡ RLC Circuit", "🧾 Unit Convert"
]
selected = st.sidebar.radio("Tool", tools)

x = sp.Symbol('x')

def plot_solution(t, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y(t)'))
    fig.update_layout(title='Solution Plot', xaxis_title='t', yaxis_title='y(t)', height=400)
    st.plotly_chart(fig, use_container_width=True)

if selected == "🔢 Integrate":
    st.header("🔢 Integral Solver")
    f_str = st.text_input("Enter function f(x)", "x**2")
    if f_str:
        try:
            integral = sp.integrate(sp.sympify(f_str), x)
            st.latex(r"\int f(x) dx = " + sp.latex(integral) + "+ C")
        except Exception as e:
            st.error(e)

elif selected == "📉 Derive":
    st.header("📉 Derivative Calculator")
    f_str = st.text_input("Enter function f(x)", "x**3")
    if f_str:
        try:
            df = sp.diff(sp.sympify(f_str), x)
            st.latex(r"\frac{d}{dx}f(x) = " + sp.latex(df))
        except Exception as e:
            st.error(e)

elif selected == "🧮 Solve Eq":
    st.header("🧮 Equation Solver")
    eqn = st.text_input("Enter equation (e.g., x**2 - 4 = 0)", "x**2 - 4 = 0")
    if "=" in eqn:
        try:
            lhs, rhs = eqn.split("=")
            sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
            st.success(f"Solutions: {sol}")
        except Exception as e:
            st.error(e)

elif selected == "📈 ODE 2nd":
    st.header("📈 Second-Order ODE")
    ode = st.text_input("Enter ODE (e.g., y'' + y)", "y'' + y")
    y = Function('y')
    try:
        eq = Eq(eval(ode.replace("y''", "y(t).diff(t,2)").replace("y'", "y(t).diff(t)")), 0)
        sol = dsolve(eq, y(t))
        st.latex("y(t) = " + sp.latex(sol.rhs))
    except Exception as e:
        st.error(e)

elif selected == "🧊 ODE 1st":
    st.header("🧊 First-Order ODE")
    f_str = st.text_input("dy/dt = f(t,y)", "-2*y + 1")
    y0 = st.number_input("Initial condition y(0)", value=1.0)
    T = st.slider("Time range [s]", 5, 50, 10)
    try:
        f = lambda t, y: eval(f_str, {"t": t, "y": y, "np": np})
        sol = solve_ivp(f, [0, T], [y0], t_eval=np.linspace(0, T, 500))
        plot_solution(sol.t, sol.y[0])
    except Exception as e:
        st.error(e)

elif selected == "⚙️ Control":
    st.header("⚙️ Control System Response")
    m = st.slider("Mass m", 0.1, 10.0, 1.0)
    c = st.slider("Damping c", 0.0, 10.0, 1.0)
    k = st.slider("Spring Constant k", 1.0, 100.0, 20.0)
    sys = tf([1], [m, c, k])
    t_vals = np.linspace(0, 10, 1000)
    t, y = step_response(sys, t_vals)
    plot_solution(t, y)

elif selected == "🧲 Lagrangian":
    st.header("🧲 Lagrangian Mechanics")
    q = Function('q')(t)
    m_sym, k_sym = sp.symbols('m k')
    L = (1/2)*m_sym*sp.diff(q, t)**2 - (1/2)*k_sym*q**2
    eq = sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)
    st.latex("L = T - V = " + sp.latex(L))
    st.latex(sp.latex(eq) + " = 0")

elif selected == "📐 Beam Forces":
    st.header("📐 Beam Reaction Forces")
    Lb = st.number_input("Beam Length L [m]", 1.0)
    F = st.number_input("Center Load F [N]", 100.0)
    st.success(f"Reaction Forces: A = {F/2:.2f} N, B = {F/2:.2f} N")

elif selected == "📊 Matrix Tools":
    st.header("📊 Matrix Tools")
    A = st.text_input("Matrix A (e.g. [[2,1],[1,3]])", "[[2,1],[1,3]]")
    b = st.text_input("Vector b (e.g. [4,5])", "[4,5]")
    try:
        A_mat = sp.Matrix(eval(A))
        b_vec = sp.Matrix(eval(b))
        st.write("x =", A_mat.LUsolve(b_vec))
        st.write("det(A) =", A_mat.det())
        st.write("Eigenvalues =", A_mat.eigenvals())
    except Exception as e:
        st.error(e)

elif selected == "🔄 Laplace":
    st.header("🔄 Laplace Transform")
    f_str = st.text_input("f(t)", "exp(-t)*sin(t)")
    try:
        F = laplace_transform(sp.sympify(f_str), t, s, noconds=True)
        st.latex(r"\mathcal{L}\{" + sp.latex(sp.sympify(f_str)) + r"\} = " + sp.latex(F))
    except Exception as e:
        st.error(e)

elif selected == "⚡ RLC Circuit":
    st.header("⚡ RLC Circuit Impedance")
    R = st.number_input("Resistance R [Ω]", 1.0)
    L_val = st.number_input("Inductance L [H]", 0.1)
    C_val = st.number_input("Capacitance C [F]", 0.01)
    freq = st.slider("Frequency [Hz]", 10, 1000, 50)
    w = 2 * math.pi * freq
    Z = complex(R, w * L_val - 1 / (w * C_val))
    st.success(f"Z = {Z.real:.2f} + j{Z.imag:.2f} Ω")

elif selected == "🧾 Unit Convert":
    st.header("🧾 Unit Converter")
    opt = st.selectbox("Convert", ["N→kgf", "kgf→N", "Pa→bar", "bar→Pa", "m/s→km/h", "km/h→m/s"])
    val = st.number_input("Value", value=1.0)
    conv = {
        "N→kgf": lambda x: x / 9.80665,
        "kgf→N": lambda x: x * 9.80665,
        "Pa→bar": lambda x: x / 1e5,
        "bar→Pa": lambda x: x * 1e5,
        "m/s→km/h": lambda x: x * 3.6,
        "km/h→m/s": lambda x: x / 3.6
    }
    st.success(f"Result: {conv[opt](val):.4f}")
