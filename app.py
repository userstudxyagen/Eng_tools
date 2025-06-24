# 🇩🇪 Modernes Engineering-Tool (vollständige Version)

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from scipy.integrate import solve_ivp
from control import tf, step_response
from sympy.abc import s, t
from sympy import Function, Eq, dsolve, laplace_transform, symbols

# Initialisierung
st.set_page_config(page_title="Ingenieur-Helfer", layout="wide")
st.title("🔧 Ingenieur-Helfer (Deutsch)")

# Hilfsfunktionen
def tool_info(title, description, usage, examples=None):
    with st.expander("ℹ️ Info"):
        st.markdown(f"**{title}**")
        st.markdown(f"**Beschreibung:** {description}")
        st.markdown(f"**Verwendung:** {usage}")
        if examples:
            st.markdown("**Beispiele:**")
            for example in examples:
                st.markdown(f"- {example}")

def plot_loesung(t, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y(t)'))
    fig.update_layout(title='Lösungsdiagramm', xaxis_title='t', yaxis_title='y(t)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/67/000000/external-engineering-industrial-engineering-flat-icons-inmotus-design.png", width=80)
st.sidebar.title("🛠️ Wähle ein Werkzeug")

werkzeuge = [
    "🔢 Integrieren", "📉 Ableiten", "🧮 Gleichung lösen", "📈 DGL 2. Ordnung",
    "🧊 DGL 1. Ordnung", "⚙️ Regelungstechnik", "🧲 Lagrange-Mechanik",
    "📐 Balkenkräfte", "📊 Matrizenwerkzeuge", "🔄 Laplace-Transformation",
    "⚡ RLC-Schaltung", "🧾 Einheitenrechner",
    "📌 Biegemomentdiagramm", "🚗 Kinematik-Rechner",
    "📚 Physikalische Formeln", "🎥 2D-Simulationen"
]
auswahl = st.sidebar.radio("Werkzeug", werkzeuge)

# Symbole definieren
x, y = symbols('x y')

# 1. Integralrechner
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
            st.error(f"Fehler: {str(e)}")

# 2. Ableitungsrechner
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
            st.error(f"Fehler: {str(e)}")

# 3. Gleichungslöser
elif auswahl == "🧮 Gleichung lösen":
    st.header("🧮 Gleichungslöser")
    tool_info(
        "Gleichungslöser",
        "Löst algebraische Gleichungen nach x auf.",
        "Geben Sie eine Gleichung in x ein (mit '=' Zeichen).",
        ["x² - 4 = 0 → x = ±2", "x³ + 2x² - 5x - 6 = 0 → x = -3, -1, 2"]
    )
    eqn = st.text_input("Gleichung eingeben (z.B. x**2 - 4 = 0)", "x**2 - 4 = 0")
    if "=" in eqn:
        try:
            lhs, rhs = eqn.split("=")
            sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
            st.success(f"Lösungen: {sol}")
        except Exception as e:
            st.error(f"Fehler: {str(e)}")

# 4. DGL 2. Ordnung
elif auswahl == "📈 DGL 2. Ordnung":
    st.header("📈 DGL 2. Ordnung")
    tool_info(
        "Differentialgleichung 2. Ordnung",
        "Löst lineare gewöhnliche Differentialgleichungen 2. Ordnung.",
        "Geben Sie die DGL in der Form y'' + y ein (mit y als Funktion von t).",
        ["y'' + y = 0 → y(t) = C₁⋅cos(t) + C₂⋅sin(t)"]
    )
    ode = st.text_input("DGL eingeben (z.B. y'' + y)", "y'' + y")
    try:
        y = Function('y')
        eq = Eq(eval(ode.replace("y''", "y(t).diff(t,2)").replace("y'", "y(t).diff(t)")), 0)
        sol = dsolve(eq, y(t))
        st.latex("y(t) = " + sp.latex(sol.rhs))
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

# 5. DGL 1. Ordnung
elif auswahl == "🧊 DGL 1. Ordnung":
    st.header("🧊 DGL 1. Ordnung")
    tool_info(
        "Differentialgleichung 1. Ordnung",
        "Löst numerisch Differentialgleichungen 1. Ordnung mit Anfangswert.",
        "Geben Sie dy/dt = f(t,y) ein und den Anfangswert y(0).",
        ["dy/dt = -2y → exponentielle Abnahme", "dy/dt = y → exponentielles Wachstum"]
    )
    f_str = st.text_input("dy/dt = f(t,y)", "-2*y + 1")
    y0 = st.number_input("Anfangswert y(0)", value=1.0)
    T = st.slider("Zeitbereich [s]", 5, 50, 10)
    try:
        f = lambda t, y: eval(f_str, {"t": t, "y": y, "np": np})
        sol = solve_ivp(f, [0, T], [y0], t_eval=np.linspace(0, T, 500))
        plot_loesung(sol.t, sol.y[0])
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

# 6. Regelungstechnik
elif auswahl == "⚙️ Regelungstechnik":
    st.header("⚙️ Regelungstechnik – Sprungantwort")
    tool_info(
        "Regelungstechnik - Sprungantwort",
        "Simuliert die Sprungantwort eines PT2-Systems (Masse-Feder-Dämpfer).",
        "Stellen Sie die Parameter m, c und k ein und sehen Sie die Systemantwort.",
        ["m=1, c=2, k=20 → gedämpfte Schwingung", "m=1, c=10, k=25 → aperiodischer Grenzfall"]
    )
    m = st.slider("Masse m", 0.1, 10.0, 1.0)
    c = st.slider("Dämpfung c", 0.0, 10.0, 1.0)
    k = st.slider("Federkonstante k", 1.0, 100.0, 20.0)
    sys = tf([1], [m, c, k])
    t_vals = np.linspace(0, 10, 1000)
    t, y = step_response(sys, t_vals)
    plot_loesung(t, y)

# 7. Lagrange-Mechanik
elif auswahl == "🧲 Lagrange-Mechanik":
    st.header("🧲 Lagrange-Mechanik")
    tool_info(
        "Lagrange-Mechanik",
        "Leitet die Bewegungsgleichungen nach der Lagrange-Formulierung her.",
        "Zeigt die Lagrange-Gleichung für ein Feder-Masse-System.",
        ["L = T - V → Bewegungsgleichung für harmonischen Oszillator"]
    )
    q = Function('q')(t)
    m_sym, k_sym = sp.symbols('m k')
    L = (1/2)*m_sym*sp.diff(q, t)**2 - (1/2)*k_sym*q**2
    eq = sp.diff(sp.diff(L, sp.diff(q, t)), t) - sp.diff(L, q)
    st.latex("L = T - V = " + sp.latex(L))
    st.latex(sp.latex(eq) + " = 0")

# 8. Balkenkräfte
elif auswahl == "📐 Balkenkräfte":
    st.header("📐 Reaktionskräfte eines Balkens")
    tool_info(
        "Balkenkräfte",
        "Berechnet die Auflagerreaktionen für einen statisch bestimmt gelagerten Balken mit Einzellast.",
        "Geben Sie Balkenlänge und Last ein, um die Auflagerkräfte zu berechnen.",
        ["L=5m, F=100N → A=B=50N"]
    )
    Lb = st.number_input("Balkenlänge L [m]", 1.0)
    F = st.number_input("Zentrische Last F [N]", 100.0)
    st.success(f"Reaktionskräfte: A = {F/2:.2f} N, B = {F/2:.2f} N")

# 9. Matrizenwerkzeuge
elif auswahl == "📊 Matrizenwerkzeuge":
    st.header("📊 Matrix-Rechner")
    tool_info(
        "Matrix-Rechner",
        "Führt Matrixoperationen durch: Lösen linearer Gleichungssysteme, Determinante, Eigenwerte.",
        "Geben Sie Matrix A und Vektor b für Ax=b ein.",
        ["A=[[2,1],[1,3]], b=[4,5] → x=[1,2]", "A=[[1,2],[3,4]] → det(A)=-2"]
    )
    A = st.text_input("Matrix A (z.B. [[2,1],[1,3]])", "[[2,1],[1,3]]")
    b = st.text_input("Vektor b (z.B. [4,5])", "[4,5]")
    try:
        A_mat = sp.Matrix(eval(A))
        b_vec = sp.Matrix(eval(b))
        st.write("Lösung x =", A_mat.LUsolve(b_vec))
        st.write("Determinante det(A) =", A_mat.det())
        st.write("Eigenwerte =", A_mat.eigenvals())
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

# 10. Laplace-Transformation
elif auswahl == "🔄 Laplace-Transformation":
    st.header("🔄 Laplace-Transformation")
    tool_info(
        "Laplace-Transformation",
        "Führt die Laplace-Transformation einer Zeitfunktion durch.",
        "Geben Sie eine Funktion f(t) ein, die transformiert werden soll.",
        ["exp(-t) → 1/(s+1)", "sin(t) → 1/(s²+1)"]
    )
    f_str = st.text_input("f(t)", "exp(-t)*sin(t)")
    try:
        F = laplace_transform(sp.sympify(f_str), t, s, noconds=True)
        st.latex(r"\mathcal{L}\{" + sp.latex(sp.sympify(f_str)) + r"\} = " + sp.latex(F))
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

# 11. RLC-Schaltung
elif auswahl == "⚡ RLC-Schaltung":
    st.header("⚡ Impedanz einer RLC-Schaltung")
    tool_info(
        "RLC-Schaltung",
        "Berechnet die komplexe Impedanz einer Serienschaltung aus R, L und C.",
        "Stellen Sie die Bauteilwerte und Frequenz ein.",
        ["R=10Ω, L=0.1H, C=0.01F, f=50Hz → Z=10 + j...Ω"]
    )
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
    tool_info(
        "Einheitenrechner",
        "Rechnet zwischen verschiedenen physikalischen Einheiten um.",
        "Wählen Sie die Umrechnungsrichtung und geben Sie den Wert ein.",
        ["100N → 10.197kgf", "1bar → 100000Pa", "10m/s → 36km/h"]
    )
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

# 13. Biegemomentdiagramm
elif auswahl == "📌 Biegemomentdiagramm":
    st.header("📌 Biegemoment-Diagramm für Einfeldträger")
    tool_info(
        "Biegemomentdiagramm",
        "Zeigt den Biegemomentverlauf für einen beidseitig gelenkig gelagerten Balken mit mittiger Einzellast.",
        "Geben Sie Balkenlänge und Last ein.",
        ["L=5m, P=100N → M_max=125Nm in der Mitte"]
    )
    L = st.number_input("Trägerlänge L [m]", 5.0)
    P = st.number_input("Einzellast P [N] (mittig)", 100.0)
    x_vals = np.linspace(0, L, 500)
    M = np.where(x_vals < L/2, P * x_vals / 2, P * (L - x_vals) / 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=M, mode='lines', name='M(x)'))
    fig.update_layout(title='Biegemomentverlauf', xaxis_title='x [m]', yaxis_title='M(x) [Nm]')
    st.plotly_chart(fig)

# 14. Kinematik-Rechner
elif auswahl == "🚗 Kinematik-Rechner":
    st.header("🚗 Kinematik-Rechner (gleichmäßig beschleunigte Bewegung)")
    tool_info(
        "Kinematik-Rechner",
        "Berechnet den Weg-Zeit-Verlauf einer gleichmäßig beschleunigten Bewegung.",
        "Geben Sie Anfangsgeschwindigkeit und Beschleunigung ein.",
        ["v0=0, a=1m/s² → s(5s)=12.5m", "v0=10m/s, a=-2m/s² → Bremsweg bis v=0: 25m"]
    )
    v0 = st.number_input("Anfangsgeschwindigkeit v₀ [m/s]", 0.0)
    a = st.number_input("Beschleunigung a [m/s²]", 1.0)
    t_end = st.slider("Zeitdauer t [s]", 1, 20, 5)
    t_vals = np.linspace(0, t_end, 200)
    s_vals = v0 * t_vals + 0.5 * a * t_vals**2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals, y=s_vals, mode='lines', name='s(t)'))
    fig.update_layout(title='Orts-Zeit-Verlauf', xaxis_title='t [s]', yaxis_title='s(t) [m]')
    st.plotly_chart(fig)

# 15. Physikalische Formeln
elif auswahl == "📚 Physikalische Formeln":
    st.header("📚 Physikalische Formelsammlung")
    tool_info(
        "Formelsammlung",
        "Durchsuche wichtige Formeln nach Stichwörtern (Mechanik, Thermo, etc.).",
        "Geben Sie einen Begriff wie 'Energie', 'Druck', 'Kraft' ein.",
        ["s = v⋅t", "F = m⋅a", "p = F / A", "Q = m⋅c⋅ΔT"]
    )

    formeln = {
        "Geschwindigkeit": r"v = \frac{s}{t}",
        "Beschleunigung": r"a = \frac{\Delta v}{\Delta t}",
        "Kraft": r"F = m \cdot a",
        "Arbeit": r"W = F \cdot s",
        "Leistung": r"P = \frac{W}{t}",
        "Druck": r"p = \frac{F}{A}",
        "Dichte": r"\rho = \frac{m}{V}",
        "Impuls": r"p = m \cdot v",
        "Kinetische Energie": r"E_{kin} = \frac{1}{2}mv^2",
        "Potenzielle Energie": r"E_{pot} = mgh",
        "Ohmsches Gesetz": r"U = R \cdot I",
        "Elektrische Leistung": r"P = U \cdot I",
        "Wärmeenergie": r"Q = m \cdot c \cdot \Delta T",
        "Wirkungsgrad": r"\eta = \frac{P_{ab}}{P_{zu}}",
        "Hooke'sches Gesetz": r"\sigma = E \cdot \varepsilon"
    }

    suchbegriff = st.text_input("🔍 Formel suchen", "").lower()
    treffer = {k: v for k, v in formeln.items() if suchbegriff in k.lower() or suchbegriff in v.lower()}

    if suchbegriff:
        if treffer:
            for name, f in treffer.items():
                st.markdown(f"**{name}:**")
                st.latex(f)
        else:
            st.warning("Keine passende Formel gefunden.")
    else:
        st.info("Bitte einen Suchbegriff eingeben.")

# 16. 2D-Simulationen
elif auswahl == "🎥 2D-Simulationen":
    st.header("🎥 2D-Simulationen für Maschinenbau")
    tool_info(
        "2D-Simulationen",
        "Visualisiert klassische Mechanik-Szenarien in 2D für das Maschinenbau-Studium.",
        "Wählen Sie eine Szene (z.B. Feder-Masse-System) und stellen Sie die Parameter ein.",
        ["z.B. Masse-Feder-Dämpfer", "z.B. Pendel", "z.B. schräge Ebene mit Reibung"]
    )

    szenen = ["Masse-Feder-Dämpfer", "Einfachpendel", "Block auf schiefer Ebene (mit Reibung)", "2D-Kollision", "Zentralkraftbewegung"]
    sz = st.selectbox("🎞️ Simulations-Szene auswählen", szenen)

    if sz == "Masse-Feder-Dämpfer":
        # Parameter
        m = st.slider("Masse m [kg]", 0.1, 10.0, 1.0)
        k = st.slider("Federkonstante k [N/m]", 1.0, 100.0, 20.0)
        c = st.slider("Dämpfung c [Ns/m]", 0.0, 10.0, 0.5)
        y0 = st.slider("Anfangsauslenkung [m]", -1.0, 1.0, 0.5)
        v0 = st.slider("Anfangsgeschwindigkeit [m/s]", -5.0, 5.0, 0.0)
        T = st.slider("Simulationsdauer [s]", 2, 20, 10)

        # DGL definieren
        def f(t, y):  # y = [s, v]
            return [y[1], (-c * y[1] - k * y[0]) / m]

        sol = solve_ivp(f, [0, T], [y0, v0], t_eval=np.linspace(0, T, 500))

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], mode="lines", name="Auslenkung y(t)"))
        fig.update_layout(title="Masse-Feder-Dämpfer Simulation", xaxis_title="Zeit [s]", yaxis_title="Auslenkung [m]")
        st.plotly_chart(fig, use_container_width=True)
