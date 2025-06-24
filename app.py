# Zuerst fügen wir eine Hilfsfunktion für die einheitliche Darstellung der Beschreibungen hinzu
def tool_info(title, description, usage, examples=None):
    with st.expander("ℹ️ Info"):
        st.markdown(f"**{title}**")
        st.markdown(f"**Beschreibung:** {description}")
        st.markdown(f"**Verwendung:** {usage}")
        if examples:
            st.markdown("**Beispiele:**")
            for example in examples:
                st.markdown(f"- {example}")

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
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

# 3. Gleichung lösen
elif auswahl == "🧮 Gleichung lösen":
    st.header("🧮 Gleichungslöser")
    tool_info(
        "Gleichungslöser",
        "Löst algebraische Gleichungen nach x auf.",
        "Geben Sie eine Gleichung in x ein (mit '=' Zeichen).",
        ["x² - 4 = 0 → x = ±2", "x³ + 2x² - 5x - 6 = 0 → x = -3, -1, 2"]
    )
    eqn = st.text_input("Gleichung eingeben (z. B. x**2 - 4 = 0)", "x**2 - 4 = 0")
    # Rest des bestehenden Codes...

# 4. DGL 2. Ordnung
elif auswahl == "📈 DGL 2. Ordnung":
    st.header("📈 DGL 2. Ordnung")
    tool_info(
        "Differentialgleichung 2. Ordnung",
        "Löst lineare gewöhnliche Differentialgleichungen 2. Ordnung.",
        "Geben Sie die DGL in der Form y'' + y ein (mit y als Funktion von t).",
        ["y'' + y = 0 → y(t) = C₁⋅cos(t) + C₂⋅sin(t)"]
    )
    ode = st.text_input("DGL eingeben (z. B. y'' + y)", "y'' + y")
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

# 7. Lagrange
elif auswahl == "🧲 Lagrange-Mechanik":
    st.header("🧲 Lagrange-Mechanik")
    tool_info(
        "Lagrange-Mechanik",
        "Leitet die Bewegungsgleichungen nach der Lagrange-Formulierung her.",
        "Zeigt die Lagrange-Gleichung für ein Feder-Masse-System.",
        ["L = T - V → Bewegungsgleichung für harmonischen Oszillator"]
    )
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

# 9. Matrizen
elif auswahl == "📊 Matrizenwerkzeuge":
    st.header("📊 Matrix-Rechner")
    tool_info(
        "Matrix-Rechner",
        "Führt Matrixoperationen durch: Lösen linearer Gleichungssysteme, Determinante, Eigenwerte.",
        "Geben Sie Matrix A und Vektor b für Ax=b ein.",
        ["A=[[2,1],[1,3]], b=[4,5] → x=[1,2]", "A=[[1,2],[3,4]] → det(A)=-2"]
    )
    A = st.text_input("Matrix A (z. B. [[2,1],[1,3]])", "[[2,1],[1,3]]")
    # Rest des bestehenden Codes...

# 10. Laplace
elif auswahl == "🔄 Laplace-Transformation":
    st.header("🔄 Laplace-Transformation")
    tool_info(
        "Laplace-Transformation",
        "Führt die Laplace-Transformation einer Zeitfunktion durch.",
        "Geben Sie eine Funktion f(t) ein, die transformiert werden soll.",
        ["exp(-t) → 1/(s+1)", "sin(t) → 1/(s²+1)"]
    )
    f_str = st.text_input("f(t)", "exp(-t)*sin(t)")
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

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
    # Rest des bestehenden Codes...

# 13. Biegemomentdiagramm (neu)
elif auswahl == "📌 Biegemomentdiagramm":
    st.header("📌 Biegemoment-Diagramm für Einfeldträger")
    tool_info(
        "Biegemomentdiagramm",
        "Zeigt den Biegemomentverlauf für einen beidseitig gelenkig gelagerten Balken mit mittiger Einzellast.",
        "Geben Sie Balkenlänge und Last ein.",
        ["L=5m, P=100N → M_max=125Nm in der Mitte"]
    )
    L = st.number_input("Trägerlänge L [m]", 5.0)
    # Rest des bestehenden Codes...

# 14. Kinematik-Rechner (neu)
elif auswahl == "🚗 Kinematik-Rechner":
    st.header("🚗 Kinematik-Rechner (gleichmäßig beschleunigte Bewegung)")
    tool_info(
        "Kinematik-Rechner",
        "Berechnet den Weg-Zeit-Verlauf einer gleichmäßig beschleunigten Bewegung.",
        "Geben Sie Anfangsgeschwindigkeit und Beschleunigung ein.",
        ["v0=0, a=1m/s² → s(5s)=12.5m", "v0=10m/s, a=-2m/s² → Bremsweg bis v=0: 25m"]
    )
    v0 = st.number_input("Anfangsgeschwindigkeit v₀ [m/s]", 0.0)
    # Rest des bestehenden Codes...
