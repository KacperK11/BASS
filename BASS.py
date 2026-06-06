
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.optimize import minimize

# =========================================================
# KONFIGURACJA
# =========================================================

st.set_page_config(
    page_title="Model Dyfuzji Bassa",
    layout="wide"
)
st.markdown("""
<style>

/* kolumny streamlit */
div[data-testid="column"] {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)
st.title("Wieloproduktowy Model Dyfuzji Bassa")

# =========================================================
# DANE EMPIRYCZNE
# =========================================================

data = pd.DataFrame({
    "Year": list(range(1999, 2026)),
    "Wind": [
        5.53, 9.35, 10.46, 15.86, 19.09, 26.02, 27.77,
        31.32, 40.51, 41.38, 39.42, 38.55, 49.86,
        51.68, 52.74, 58.5, 80.62, 79.92, 105.69,
        109.95, 125.89, 132.11, 114.16, 124.81,
        137.51, 138.57, 133
    ],
    "Solar": [
        0.02, 0, 0.1, 0.2, 0.31, 0.56, 1.31,
        2.27, 3.14, 4.51, 6.72, 11.96, 19.99,
        26.74, 30.62, 35.45, 38.08, 37.56, 38.76,
        44.32, 45.22, 49.5, 49.34, 60.3,
        63.87, 74.13, 89.62
    ]
})

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "Klasyczny Bass",
    "Model dwuproduktowy",
    "Wind vs Solar"
])



# =========================================================
# TAB 1 — KLASYCZNY MODEL BASSA
# =========================================================

with tab1:

    st.header("Klasyczny model Bassa")

    # =====================================================
    # 1. WPROWADZENIE I ZAŁOŻENIA
    # =====================================================

    with st.expander("Wprowadzenie i założenia modelu", expanded=False):

        st.markdown("""
## Wprowadzenie

Model dyfuzji Bassa został zaproponowany przez
Franka M. Bassa (1969) jako model opisujący
proces adopcji nowych technologii i produktów.

Model analizuje:

- tempo adopcji,
- rozwój rynku,
- dyfuzję innowacji,
- przejście od innowatorów do imitatorów.

Model znajduje zastosowanie w:

- prognozowaniu sprzedaży,
- analizie technologii,
- ekonomii innowacji,
- modelowaniu rynków technologicznych.
""")

        st.subheader("Założenia modelu")

        assumptions_df = pd.DataFrame({

            "Założenie": [

                "Skończony potencjał rynku",

                "Jednorazowa adopcja",

                "Jednorodna populacja",

                "Brak konkurencyjnych technologii",

                "Efekt społeczny",

                "Stałe parametry"

            ],

            "Interpretacja": [

                "Istnieje maksymalna liczba użytkowników m",

                "Każdy użytkownik adoptuje technologię tylko raz",

                "Wszyscy użytkownicy podlegają tym samym parametrom",

                "Model klasyczny analizuje pojedynczą technologię",

                "Adopcja zależy od wcześniejszych użytkowników",

                "p, q oraz m są stałe w czasie"
            ]
        })

        st.dataframe(
            assumptions_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Ograniczenia modelu")

        limitations_df = pd.DataFrame({

            "Ograniczenie": [

                "Brak heterogeniczności użytkowników",

                "Stałe parametry w czasie",

                "Brak szoków zewnętrznych",

                "Brak konkurencyjnych technologii",

                "Jednorazowa adopcja"

            ],

            "Znaczenie": [

                "Model zakłada identyczne zachowania",

                "p i q nie zmieniają się dynamicznie",

                "Model nie uwzględnia kryzysów i polityki",

                "Model klasyczny analizuje jedną technologię",

                "Brak ponownych zakupów"
            ]
        })

        st.dataframe(
            limitations_df,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
    # 2. PARAMETRY MODELU
    # =====================================================

    with st.expander("Parametry modelu", expanded=False):

        st.subheader("Charakterystyka parametrów modelu")

        params_full_df = pd.DataFrame({

            "Parametr": [

                "p",

                "q",

                "m"

            ],

            "Znaczenie": [

                "Współczynnik innowacyjności",

                "Współczynnik imitacji",

                "Potencjał rynku"

            ],

            "Interpretacja ekonomiczna": [

                "Wpływ marketingu, reklamy oraz innowatorów",

                "Efekt społeczny i mechanizm word-of-mouth",

                "Maksymalna liczba potencjalnych użytkowników"

            ],

            "Ograniczenia": [

                "p ≥ 0",

                "q ≥ 0 oraz zazwyczaj q > p",

                "m > 0"

            ],

            "Wpływ na model": [

                "Wyższe p przyspiesza początkową adopcję",

                "Wyższe q wzmacnia efekt wirusowy i krzywą S",

                "Wyższe m zwiększa poziom nasycenia rynku"

            ]

        })

        st.dataframe(
            params_full_df,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
    # 3. STRUKTURA MATEMATYCZNA MODELU
    # =====================================================

    with st.expander("Struktura matematyczna modelu", expanded=False):

        st.subheader("Kompletny model Bassa")

        st.latex(r"""
f(t)=\frac{dF(t)}{dt}=(p+qF(t))(1-F(t))
""")

        st.latex(r"""
F(t)=
\frac{
1-\exp(-(p+q)t)
}{
1+\frac{q}{p}\exp(-(p+q)t)
}
""")
        st.latex(r"""
Y(t)=F(t)\cdot m
""")

        st.latex(r"""
S(t)=m(p+qF(t))(1-F(t))
""")

        st.latex(r"""
h(t)=p+qF(t)
""")

        st.latex(r"""
T^*=\frac{1}{p+q}\ln\left(\frac{q}{p}\right)
""")

                # =====================================================
        # TABELA FUNKCJI
        # =====================================================

        st.subheader("Funkcje modelu")

        functions_df = pd.DataFrame({

            "Skrót": [

                "F(t)",
                "f(t)",
                "Y(t)",
                "S(t)",
                "h(t)",
                "T*"

            ],

            "Nazwa funkcji": [

                "Skumulowany poziom adopcji",

                "Tempo adopcji",

                "Łączna liczba użytkowników",

                "Nowe adopcje / sprzedaż",

                "Funkcja hazardu adopcji",

                "Punkt przegięcia krzywej S"

            ],

            "Interpretacja i znaczenie": [

                "Określa udział rynku, który przyjął technologię do czasu t. "
                "Podstawowa funkcja opisująca proces dyfuzji i poziom nasycenia rynku.",

                "Opisuje szybkość rozprzestrzeniania się innowacji w czasie. "
                "Miernik intensywności adopcji.",

                "Przedstawia całkowitą liczbę użytkowników technologii do czasu t. "
                "Funkcja wyrażona w liczbie użytkowników lub skali rynku.",

                "Opisuje liczbę nowych adopcji w danym okresie. "
                "Pokazuje bieżącą dynamikę rynku i sprzedaży.",

                "Określa prawdopodobieństwo adopcji technologii w czasie t. "
                "Łączy efekt innowacji i wpływu społecznego.",

                "Wyznacza moment największej dynamiki wzrostu rynku "
                "oraz przejście do fazy nasycenia."

            ]

        })

        st.dataframe(
            functions_df,
            use_container_width=True,
            hide_index=True
        )

        

    # =====================================================
    # 4. DASHBOARD ANALITYCZNY
    # =====================================================

    st.subheader("Dashboard analityczny")

    # =====================================================
    # SUWAKI
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        p = st.slider(
            "p — innowacja",
            0.001,
            0.10,
            0.03,
            0.001,
            key="dashboard_p"
        )

    with col2:

        q = st.slider(
            "q — imitacja",
            0.01,
            1.00,
            0.38,
            0.01,
            key="dashboard_q"
        )

    with col3:

        m = st.slider(
            "m — potencjał rynku",
            100,
            10000,
            1000,
            100,
            key="dashboard_m"
        )

    with col4:

        T = st.slider(
            "Horyzont czasu",
            10,
            80,
            40,
            key="dashboard_T"
        )

    # =====================================================
    # SYMULACJA
    # =====================================================

    dt = 0.1

    t = np.arange(0, T, dt)

    F = np.zeros(len(t))

    F[0] = 0.001

    for i in range(1, len(t)):

        dF = (
            (p + q * F[i - 1])
            * (1 - F[i - 1])
        )

        F[i] = F[i - 1] + dF * dt

        F[i] = np.clip(F[i], 0, 1)

    Y = m * F

    f = np.gradient(F, dt)

    S = np.gradient(Y, dt)

    S = np.clip(S, 0, None)

    hazard = p + q * F

    t_star = np.log(q / p) / (p + q)

    peak_idx = np.argmax(S)

    peak_time = t[peak_idx]

    peak_sales = S[peak_idx]

    # =====================================================
    # METRYKI
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Punkt przegięcia",
            f"{t_star:.2f}"
        )

    with c2:

        st.metric(
            "Peak sales",
            f"{peak_sales:.2f}"
        )

    with c3:

        st.metric(
            "Final adoption",
            f"{Y[-1]:.0f}"
        )

        # =====================================================
    # WYKRESY — RZĄD 1
    # =====================================================

    fig_bass = go.Figure()

    fig_bass.add_trace(
        go.Scatter(
            x=t,
            y=Y,
            mode="lines",
            name="Y(t) — adopcja"
        )
    )

    # LINIA PUNKTU PRZEGIĘCIA
    fig_bass.add_vline(
        x=t_star,
        line_dash="dash",
        line_color="black"
    )

    fig_bass.update_layout(
        title="Krzywa adopcji — Y(t)",
        xaxis_title="Czas",
        yaxis_title="Adopcja ",
        height=350
    )

    # =====================================================

    fig_hazard = go.Figure()

    fig_hazard.add_trace(
        go.Scatter(
            x=t,
            y=hazard,
            mode="lines",
            name="h(t) — hazard"
        )
    )

    # LINIA PUNKTU PRZEGIĘCIA
    fig_hazard.add_vline(
        x=t_star,
        line_dash="dash",
        line_color="black"
    )

    fig_hazard.update_layout(
        title="Funkcja hazardu — h(t)",
        xaxis_title="Czas",
        yaxis_title="h(t)",
        height=350
    )

    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            fig_bass,
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            fig_hazard,
            use_container_width=True
        )

    # =====================================================
    # WYKRESY — RZĄD 2
    # =====================================================

    fig_sales = go.Figure()

    fig_sales.add_trace(
        go.Scatter(
            x=t,
            y=S,
            mode="lines",
            name="S(t) — sprzedaż"
        )
    )

    # LINIA PUNKTU PRZEGIĘCIA
    fig_sales.add_vline(
        x=t_star,
        line_dash="dash",
        line_color="black"
    )

    fig_sales.update_layout(
        title="Nowe adopcje / sprzedaż — S(t)",
        xaxis_title="Czas",
        yaxis_title="Sprzedaż",
        height=350
    )

    # =====================================================

    fig_f = go.Figure()

    fig_f.add_trace(
        go.Scatter(
            x=t,
            y=f,
            mode="lines",
            name="f(t) — tempo adopcji"
        )
    )

    # LINIA PUNKTU PRZEGIĘCIA
    fig_f.add_vline(
        x=t_star,
        line_dash="dash",
        line_color="black"
    )

    fig_f.update_layout(
        title="Tempo adopcji — f(t)",
        xaxis_title="Czas",
        yaxis_title="f(t)",
        height=350
    )

    # =====================================================

    col3, col4 = st.columns(2)

    with col3:

        st.plotly_chart(
            fig_sales,
            use_container_width=True
        )

    with col4:

        st.plotly_chart(
            fig_f,
            use_container_width=True
        )

    
    # =====================================================
    # 5. DYNAMIKA SPOŁECZNA
    # =====================================================

    with st.expander("Dynamika społeczna i interpretacja ekonomiczna", expanded=False):

        st.subheader("Kategorie adopcji Rogersa")

        st.markdown("""
    Rozkład kategorii adopcji zależy od relacji:

    - innowacji (p),
    - imitacji społecznej (q).

    Wyższe q/p oznacza silniejszy efekt społeczny
    oraz większy udział późnych adopcji.
    """)

        # =====================================================
        # SUWAKI
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:

            p_r = st.slider(
                "p — innowacja",
                0.001,
                0.10,
                0.03,
                0.001,
                key="rogers_p"
            )

        with col2:

            q_r = st.slider(
                "q — imitacja",
                0.01,
                1.00,
                0.38,
                0.01,
                key="rogers_q"
            )

        # =====================================================
        # DYNAMICZNY ROGERS
        # =====================================================

        ratio = q_r / p_r

            # ograniczenie wpływu q/p
        social_effect = min(ratio, 15)

        # dynamiczne udziały

        innowatorzy = max(
            2,
            12 - 0.5 * social_effect
        )

        wczesni_adopterzy = max(
            8,
            18 - 0.4 * social_effect
        )

        wczesna_wiekszosc = min(
            40,
            30 + 0.3 * social_effect
        )

        pozna_wiekszosc = min(
            40,
            25 + 0.4 * social_effect
        )

        # reszta populacji
        maruderzy = 100 - (
            innowatorzy
            + wczesni_adopterzy
            + wczesna_wiekszosc
            + pozna_wiekszosc
        )

    # zabezpieczenie
        maruderzy = max(2, maruderzy)

        rogers_labels = [

            "Innowatorzy",

            "Wcześni\nadopterzy",

            "Wczesna\nwiększość",

            "Późna\nwiększość",

            "Maruderzy"
        ]

        rogers_values = [

            innowatorzy,

            wczesni_adopterzy,

            wczesna_wiekszosc,

            pozna_wiekszosc,

            maruderzy
        ]

        fig_rogers = go.Figure()

        fig_rogers.add_trace(
            go.Bar(

                x=rogers_labels,

                y=rogers_values,

                text=[
                    f"{v:.1f}%"
                    for v in rogers_values
                ],

                textposition="outside"
            )
        )

        fig_rogers.update_layout(

            title="Dynamiczne kategorie adopcji",

            yaxis_title="% populacji",

            height=500
        )

        st.plotly_chart(
            fig_rogers,
            use_container_width=True
        )

        # =====================================================
        # INTERPRETACJA
        # =====================================================

        st.subheader("Interpretacja ekonomiczna")

        st.markdown(
            f"### Relacja q/p = {ratio:.2f}"
        )

        col1, col2, col3 = st.columns(3)

        # =====================================================
        # NISKI q/p
        # =====================================================

        with col1:

            st.markdown("""
        ### Niski udział q/p

        - większe znaczenie innowatorów,
        - adopcja napędzana marketingiem,
        - słabszy efekt społeczny,
        - szybsza adopcja początkowa.
        """)

        # =====================================================
        # WYSOKI q/p
        # =====================================================

        with col2:

            st.markdown("""
        ### Wysoki udział q/p

        - dominacja imitacji społecznej,
        - silny efekt word-of-mouth,
        - bardziej wyraźna krzywa S,
        - większy udział późnej większości.
        """)

        # =====================================================
        # WNIOSKI
        # =====================================================

        with col3:

            st.markdown("""
        ### Wnioski

        - wysokie p → szybszy początek rynku,
        - wysokie q → efekt wirusowy,
        - wysokie q/p → adopcja społeczna,
        - niskie q/p → dominacja innowatorów.
        """)

# =========================================================
# TAB 2 — MODEL DWUPRODUKTOWY
# =========================================================

with tab2:

    st.header("Dwuproduktowy model Bassa")

    # =====================================================
    # 1. WPROWADZENIE
    # =====================================================

    with st.expander("Rozszerzenie modelu klasycznego", expanded=False):

        st.markdown("""
## Model konkurujących technologii

Dwuproduktowy model Bassa rozszerza model klasyczny
o możliwość analizy:

- konkurencji między technologiami,
- substytucji produktów,
- efektów sieciowych,
- wzajemnego wpływu adopcji.

Model opisuje sytuację, w której:

- adopcja jednego produktu
może wzmacniać lub osłabiać
adopcję drugiego produktu.

---

## Interpretacja ekonomiczna

Model może opisywać np.:

- Android vs iOS,
- VHS vs Betamax,
- Netflix vs Disney+,
- technologie energetyczne,
- konkurencyjne platformy cyfrowe.
""")

    # =====================================================
    # 2. PARAMETRY MODELU
    # =====================================================

    with st.expander("Parametry interakcji", expanded=False):

        params_dual_df = pd.DataFrame({

            "Parametr": [

                "a₁₂",

                "a₂₁"

            ],

            "Znaczenie": [

                "Wpływ produktu 2 na produkt 1",

                "Wpływ produktu 1 na produkt 2"

            ],

            "Interpretacja ekonomiczna": [

                "Substytucja lub wsparcie adopcji produktu 1",

                "Substytucja lub wsparcie adopcji produktu 2"

            ],

            "Efekt": [

                "a₁₂ > 0 → wsparcie, a₁₂ < 0 → konkurencja",

                "a₂₁ > 0 → wsparcie, a₂₁ < 0 → konkurencja"
            ]
        })

        st.dataframe(
            params_dual_df,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 3. STRUKTURA MATEMATYCZNA
    # =====================================================


    with st.expander("Struktura matematyczna modelu", expanded=False):

        st.subheader("Układ równań modelu")

        # =====================================================
        # RÓWNANIA DYNAMIKI
        # =====================================================

        st.latex(r"""
        f_1(t)=\frac{dF_1(t)}{dt}=
        (p_1+q_1F_1(t)+a_{12}F_2(t))(1-F_1(t))
        """)

        st.latex(r"""
        f_2(t)=\frac{dF_2(t)}{dt}=
        (p_2+q_2F_2(t)+a_{21}F_1(t))(1-F_2(t))
        """)

        # =====================================================
        # POZIOM ADOPCJI
        # =====================================================

        st.latex(r"""
        Y_1(t)=m_1F_1(t)
        """)

        st.latex(r"""
        Y_2(t)=m_2F_2(t)
        """)

        # =====================================================
        # NOWE ADOPCJE
        # =====================================================

        st.latex(r"""
        S_1(t)=\frac{dY_1(t)}{dt}
        """)

        st.latex(r"""
        S_2(t)=\frac{dY_2(t)}{dt}
        """)

        # =====================================================
        # HAZARD ADOPCJI
        # =====================================================

        st.latex(r"""
        h_1(t)=
        p_1+q_1F_1(t)+a_{12}F_2(t)
        """)

        st.latex(r"""
        h_2(t)=
        p_2+q_2F_2(t)+a_{21}F_1(t)
        """)

        # =====================================================
        # TEMPO ADOPCJI
        # =====================================================

        st.latex(r"""
       F_1(t)=
\frac{
1-\exp\left(-(p_1+q_1+a_{12})t\right)
}{
1+\left(\frac{q_1+a_{12}}{p_1}\right)
\exp\left(-(p_1+q_1+a_{12})t\right)
}
        """)

        st.latex(r"""
        F_2(t)=
\frac{
1-\exp\left(-(p_2+q_2+a_{21})t\right)
}{
1+\left(\frac{q_2+a_{21}}{p_2}\right)
\exp\left(-(p_2+q_2+a_{21})t\right)
}
        """)

        # =====================================================
        # PUNKT PRZEGIĘCIA
        # =====================================================

        st.latex(r"""
        T_1^*=
        \frac{1}{p_1+q_1}
        \ln\left(
        \frac{q_1+a_{12}}{p_1}
        \right)
        """)

        st.latex(r"""
        T_2^*=
        \frac{1}{p_2+q_2}
        \ln\left(
        \frac{q_2+a_{21}}{p_2}
        \right)
        """)

        # =====================================================
        # OPIS
        # =====================================================

        st.markdown("""
    Model opisuje jednoczesną dyfuzję dwóch technologii.

    Adopcja każdego produktu zależy od:

    - innowacji,
    - imitacji,
    - efektów sieciowych,
    - wpływu konkurencyjnej technologii.
    """)

        # =====================================================
        # TABELA INTERPRETACJI
        # =====================================================

        


    # =====================================================
    # 4. DASHBOARD ANALITYCZNY
    # =====================================================

    with st.expander("Dashboard analityczny", expanded=True):

        # =====================================================
        # SUWAKI
        # =====================================================

        st.subheader("Parametry symulacji")

        c1, c2, c3 = st.columns(3)

        with c1:

            p1 = st.slider(
                "p₁",
                0.001,
                0.10,
                0.03,
                0.001,
                key="p1"
            )

            q1 = st.slider(
                "q₁",
                0.01,
                1.00,
                0.38,
                0.01,
                key="q1"
            )

            m1 = st.slider(
                "m₁",
                100,
                10000,
                4000,
                100,
                key="m1"
            )

        with c2:

            p2 = st.slider(
                "p₂",
                0.001,
                0.10,
                0.02,
                0.001,
                key="p2"
            )

            q2 = st.slider(
                "q₂",
                0.01,
                1.00,
                0.45,
                0.01,
                key="q2"
            )

            m2 = st.slider(
                "m₂",
                100,
                10000,
                5000,
                100,
                key="m2"
            )

        with c3:

            a12 = st.slider(
                "a₁₂",
                -1.0,
                1.0,
                -0.2,
                0.01,
                key="a12"
            )

            a21 = st.slider(
                "a₂₁",
                -1.0,
                1.0,
                -0.1,
                0.01,
                key="a21"
            )

            T2 = st.slider(
                "Horyzont czasu",
                10,
                80,
                40,
                key="T2"
            )
            # =====================================================
        # OSTRZEŻENIA STABILNOŚCI
        # =====================================================

        if abs(a12) > q1:

            st.warning(
                "a₁₂ przekracza q₁ — możliwa niestabilność modelu."
            )

        if abs(a21) > q2:

            st.warning(
                "a₂₁ przekracza q₂ — możliwa niestabilność modelu."
            )


        # =====================================================
        # SYMULACJA
        # =====================================================

        dt = 0.1

        t2 = np.arange(0, T2, dt)

        F1 = np.zeros(len(t2))
        F2 = np.zeros(len(t2))

        F1[0] = 0.001
        F2[0] = 0.001

        for i in range(1, len(t2)):

            dF1 = (
                (
                    p1
                    + q1 * F1[i - 1]
                    + a12 * F2[i - 1]
                )
                * (1 - F1[i - 1])
            )

            dF2 = (
                (
                    p2
                    + q2 * F2[i - 1]
                    + a21 * F1[i - 1]
                )
                * (1 - F2[i - 1])
            )

            F1[i] = F1[i - 1] + dF1 * dt
            F2[i] = F2[i - 1] + dF2 * dt

            F1[i] = np.clip(F1[i], 0, 1)
            F2[i] = np.clip(F2[i], 0, 1)

        Y1 = m1 * F1
        Y2 = m2 * F2

        S1 = np.gradient(Y1, dt)
        S2 = np.gradient(Y2, dt)

        S1 = np.clip(S1, 0, None)
        S2 = np.clip(S2, 0, None)

        share1 = Y1 / (Y1 + Y2 + 1e-9)
        share2 = Y2 / (Y1 + Y2 + 1e-9)

            # punkt przegięcia = maksimum sprzedaży
        peak_idx1 = np.argmax(S1)
        peak_idx2 = np.argmax(S2)

        t_star1 = t2[peak_idx1]
        t_star2 = t2[peak_idx2]

        # =====================================================
        # METRYKI
        # =====================================================

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            st.metric(
                "Final adoption P1",
                f"{Y1[-1]:.0f}"
            )

        with mcol2:
            st.metric(
                "Final adoption P2",
                f"{Y2[-1]:.0f}"
            )

        with mcol3:
            st.metric(
                "Peak sales P1",
                f"{S1.max():.2f}"
            )

        with mcol4:
            st.metric(
                "Peak sales P2",
                f"{S2.max():.2f}"
            )

        

        # =====================================================
        # WYKRESY — RZĄD 1
        # =====================================================

        fig_y1 = go.Figure()

        fig_y1.add_vline(
            x=t_star1,
            line_dash="dash",
            line_color="black"
        )

        fig_y1.add_trace(
            go.Scatter(
                x=t2,
                y=Y1,
                mode="lines",
                name="Produkt 1"
            )
        )

        fig_y1.update_layout(
            title="Adopcja produktu 1 — Y₁(t)",
            xaxis_title="Czas",
            yaxis_title="Liczba użytkowników",
            height=320
        )

        # =====================================================

        fig_y2 = go.Figure()

        fig_y2.add_vline(
            x=t_star2,
            line_dash="dash",
            line_color="black"
        )

        fig_y2.add_trace(
            go.Scatter(
                x=t2,
                y=Y2,
                mode="lines",
                name="Produkt 2"
            )
        )

        fig_y2.update_layout(
            title="Adopcja produktu 2 — Y₂(t)",
            xaxis_title="Czas",
            yaxis_title="Liczba użytkowników",
            height=320
        )

        # =====================================================

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                fig_y1,
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                fig_y2,
                use_container_width=True
            )

        # =====================================================
        # WYKRESY — RZĄD 2
        # =====================================================

        fig_s1 = go.Figure()

        fig_s1.add_trace(
            go.Scatter(
                x=t2,
                y=S1,
                mode="lines",
                name="Sprzedaż P1"
            )
        )

        fig_s1.add_vline(
            x=t_star1,
            line_dash="dash",
            line_color="black"
        )

        fig_s1.update_layout(
            title="Sprzedaż produktu 1 — S₁(t)",
            xaxis_title="Czas",
            yaxis_title="Sprzedaż okresowa",
            height=320
        )

        # =====================================================

        fig_s2 = go.Figure()

        fig_s2.add_trace(
            go.Scatter(
                x=t2,
                y=S2,
                mode="lines",
                name="Sprzedaż P2"
            )
        )

        fig_s2.add_vline(
            x=t_star2,
            line_dash="dash",
            line_color="black"
        )

        fig_s2.update_layout(
            title="Sprzedaż produktu 2 — S₂(t)",
            xaxis_title="Czas",
            yaxis_title="Sprzedaż okresowa",
            height=320
        )

        # =====================================================

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                fig_s1,
                use_container_width=True
            )

        with col4:
            st.plotly_chart(
                fig_s2,
                use_container_width=True
            )

        # =====================================================
        # WYKRESY — RZĄD 3
        # =====================================================

        fig_share = go.Figure()

        fig_share.add_trace(
            go.Scatter(
                x=t2,
                y=share1,
                mode="lines",
                name="Produkt 1"
            )
        )

        fig_share.add_trace(
            go.Scatter(
                x=t2,
                y=share2,
                mode="lines",
                name="Produkt 2"
            )
        )

        fig_share.update_layout(
            title="Udziały rynkowe",
            xaxis_title="Czas",
            yaxis_title="Udział rynku",
            height=320
        )

        # =====================================================

        interaction = a12 * F2 + a21 * F1

        fig_interaction = go.Figure()

        fig_interaction.add_trace(
            go.Scatter(
                x=t2,
                y=interaction,
                mode="lines",
                name="Interakcja"
            )
        )

        fig_interaction.update_layout(
            title="Efekt interakcji między produktami",
            xaxis_title="Czas",
            yaxis_title="Siła interakcji",
            height=320
        )

        # =====================================================

        col5, col6 = st.columns(2)

        with col5:
            st.plotly_chart(
                fig_share,
                use_container_width=True
            )

        with col6:
            st.plotly_chart(
                fig_interaction,
                use_container_width=True
            )

    # =====================================================
    # 5. INTERPRETACJA EKONOMICZNA
    # =====================================================

    with st.expander("Interpretacja ekonomiczna", expanded=False):

        st.markdown("## Konkurencja technologii")

        col1, col2, col3 = st.columns(3)

        # =====================================================
        # PARAMETRY INTERAKCJI
        # =====================================================

        with col1:

            st.markdown("""
    ### Parametry interakcji

    #### a₁₂ < 0
    - produkt 2 ogranicza adopcję produktu 1,
    - efekt konkurencji.

    #### a₁₂ > 0
    - produkt 2 wspiera adopcję produktu 1,
    - efekt komplementarności.
    """)

        # =====================================================
        # EFEKTY EKONOMICZNE
        # =====================================================

        with col2:

            st.markdown("""
    ### Efekty ekonomiczne

    Model pozwala analizować:

    - wojny technologiczne,
    - dominację rynku,
    - przewagę pierwszego gracza,
    - efekty sieciowe,
    - konkurencję platform cyfrowych.
    """)

        # =====================================================
        # WNIOSKI
        # =====================================================

        with col3:

            st.markdown("""
    ### Wnioski

    - wysokie q → silniejszy efekt społeczny,
    - wysokie a₁₂/a₂₁ → silniejsze interakcje,
    - ujemne interakcje → substytucja,
    - dodatnie interakcje → komplementarność.
    """)

# =========================================================
# TAB 4 — WIND VS SOLAR
# =========================================================

with tab3:

    st.header("Wind vs Solar — analiza empiryczna")

    # =====================================================
    # DANE HISTORYCZNE
    # =====================================================

    with st.expander("Dane historyczne", expanded=True):

        st.subheader("Historyczne dane adopcji")

        st.dataframe(
            data,
            use_container_width=True
        )

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Wind"],
                mode="lines+markers",
                name="Wind"
            )
        )

        fig_hist.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Solar"],
                mode="lines+markers",
                name="Solar"
            )
        )

        fig_hist.update_layout(
            title="Historyczne dane adopcji",
            xaxis_title="Rok",
            yaxis_title="Produkcja energii",
            height=500
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

        # =====================================================
    # POSTAĆ DYSKRETNA MODELU
    # =====================================================

    with st.expander(
    "Postać dyskretna dwuproduktowego modelu Bassa",
    expanded=False
):

        st.latex(r"""
        F_1(t)=\frac{Y_1(t)}{m_1}
        """)

        st.latex(r"""
        F_2(t)=\frac{Y_2(t)}{m_2}
        """)

        st.latex(r"""
        f_1(t)
        =
        F_1(t)-F_1(t-1)
        =
        \frac{S_1(t)}{m_1}
        """)

        st.latex(r"""
        f_2(t)
        =
        F_2(t)-F_2(t-1)
        =
        \frac{S_2(t)}{m_2}
        """)

        st.latex(r"""
        S_{1}(t)
        =
        p_1m_1
        +
        (q_1-p_1)Y_1(t)
        -
        \frac{q_1}{m_1}Y_1(t)^2
        +
        a_{12}Y_2(t)
        """)

        st.latex(r"""
        S_{2}(t)
        =
        p_2m_2
        +
        (q_2-p_2)Y_2(t)
        -
        \frac{q_2}{m_2}Y_2(t)^2
        +
        a_{21}Y_1(t)
        """)

        st.latex(r"""
        Y_1(t)=Y_1(t-1)+S_1(t)
        """)

        st.latex(r"""
        Y_2(t)=Y_2(t-1)+S_2(t)
        """)

        st.latex(r"""
        h_1(t)
        =
        p_1
        +
        q_1F_1(t)
        +
        a_{12}F_2(t)
        """)

        st.latex(r"""
        h_2(t)
        =
        p_2
        +
        q_2F_2(t)
        +
        a_{21}F_1(t)
        """)

        st.latex(r"""
        T_1^{*}
        =
        \frac{1}{p_1+q_1}
        \ln
        \left(
        \frac{q_1+a_{12}}{p_1}
        \right)
        """)

        st.latex(r"""
        T_2^{*}
        =
        \frac{1}{p_2+q_2}
        \ln
        \left(
        \frac{q_2+a_{21}}{p_2}
        \right)
        """)

    # =====================================================
    # PARAMETRY Z R
    # =====================================================

    p_wind = 0.0000
    q_wind = 0.2771
    m_wind = 358.70
    a_wind = -0.2870

    p_solar = 0.0071
    q_solar = 0.3307
    m_solar = 268.86
    a_solar = -0.0844

    # =====================================================
    # MODEL PRZYROSTÓW
    # =====================================================

    wind_model = (
        p_wind * m_wind
        + (q_wind - p_wind) * data["Wind"]
        - (q_wind / m_wind) * data["Wind"]**2
        + a_wind * data["Solar"]
    )

    solar_model = (
        p_solar * m_solar
        + (q_solar - p_solar) * data["Solar"]
        - (q_solar / m_solar) * data["Solar"]**2
        + a_solar * data["Wind"]
    )

    # =====================================================
    # PRZYROSTY RZECZYWISTE
    # =====================================================

    data["S_wind"] = data["Wind"].diff()
    data["S_solar"] = data["Solar"].diff()

    # =====================================================
    # REKONSTRUKCJA POZIOMÓW ADOPCJI
    # =====================================================

    wind_fit = [data["Wind"].iloc[0]]
    solar_fit = [data["Solar"].iloc[0]]

    for i in range(1, len(data)):

        wind_next = (
            wind_fit[-1]
            + wind_model.iloc[i]
        )

        solar_next = (
            solar_fit[-1]
            + solar_model.iloc[i]
        )

        wind_fit.append(wind_next)
        solar_fit.append(solar_next)

    wind_fit = np.array(wind_fit)
    solar_fit = np.array(solar_fit)

    # =====================================================
    # RESZTY
    # =====================================================

    residuals_wind = (
        data["S_wind"] - wind_model
    )

    residuals_solar = (
        data["S_solar"] - solar_model
    )

    residuals_wind = residuals_wind.fillna(0)
    residuals_solar = residuals_solar.fillna(0)

    # =====================================================
    # METRYKI
    # =====================================================

    sse = (
        np.sum(residuals_wind**2)
        + np.sum(residuals_solar**2)
    )

    rmse_wind = np.sqrt(
        np.mean(residuals_wind**2)
    )

    rmse_solar = np.sqrt(
        np.mean(residuals_solar**2)
    )

    # =====================================================
    # TABS
    # =====================================================

    fit_tab, scenario_tab = st.tabs([
        "Dopasowanie modelu",
        "Scenariusze"
    ])

    # =====================================================
    # FIT TAB
    # =====================================================

    with fit_tab:

        st.subheader("Oszacowane parametry z R")

        params_df = pd.DataFrame({

            "Parametr": [
                "p_wind",
                "q_wind",
                "m_wind",
                "a_wind",
                "p_solar",
                "q_solar",
                "m_solar",
                "a_solar"
            ],

            "Wartość": [

                p_wind,
                q_wind,
                m_wind,
                a_wind,

                p_solar,
                q_solar,
                m_solar,
                a_solar
            ]
        })

        st.dataframe(
            params_df,
            use_container_width=True,
            hide_index=True
        )

        # =====================================================
        # REKONSTRUKCJA POZIOMÓW ADOPCJI
        # =====================================================

        wind_fit = [data["Wind"].iloc[0]]
        solar_fit = [data["Solar"].iloc[0]]

        for i in range(1, len(data)):

            wind_next = (
                wind_fit[-1]
                + wind_model.iloc[i]
            )

            solar_next = (
                solar_fit[-1]
                + solar_model.iloc[i]
            )

            wind_fit.append(wind_next)
            solar_fit.append(solar_next)

        wind_fit = np.array(wind_fit)
        solar_fit = np.array(solar_fit)

        # =====================================================
        # METRYKI
        # =====================================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "SSE",
                f"{sse:.2f}"
            )

        with c2:
            st.metric(
                "RMSE Wind",
                f"{rmse_wind:.2f}"
            )

        with c3:
            st.metric(
                "RMSE Solar",
                f"{rmse_solar:.2f}"
            )

            
        # =====================================================
        # PEŁNA ANALIZA RÓWNAŃ MODELU
        # =====================================================

        st.markdown("---")
        st.subheader("Równania modelu dla danych empirycznych")

        # =====================================================
        # F1(t) i F2(t)
        # =====================================================

        F1_emp = data["Wind"] / m_wind
        F2_emp = data["Solar"] / m_solar

        fig_F = go.Figure()

        fig_F.add_trace(
            go.Scatter(
                x=data["Year"],
                y=F1_emp,
                mode="lines+markers",
                name="F₁(t) — Wind"
            )
        )

        fig_F.add_trace(
            go.Scatter(
                x=data["Year"],
                y=F2_emp,
                mode="lines+markers",
                name="F₂(t) — Solar"
            )
        )

        fig_F.update_layout(
            title="Funkcje adopcji- F(t)",
            xaxis_title="Rok",
            yaxis_title="F(t)",
            height=320
        )

        # =====================================================
        # Y1(t) i Y2(t)
        # =====================================================

        fig_Y = go.Figure()

        fig_Y.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Wind"],
                mode="lines+markers",
                name="Y₁(t) — Wind"
            )
        )

        fig_Y.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Solar"],
                mode="lines+markers",
                name="Y₂(t) — Solar"
            )
        )

        fig_Y.update_layout(
            title="Skumulowana adopcja- Y(t)",
            xaxis_title="Rok",
            yaxis_title="Produkcja energii",
            height=320
        )

        # =====================================================
        # WYŚWIETLENIE 1
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                fig_F,
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                fig_Y,
                use_container_width=True
            )

        # =====================================================
        # S1(t) i S2(t)
        # =====================================================

        fig_S = go.Figure()

        fig_S.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["S_wind"],
                mode="lines+markers",
                name="S₁(t) — Wind"
            )
        )

        fig_S.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["S_solar"],
                mode="lines+markers",
                name="S₂(t) — Solar"
            )
        )

        fig_S.update_layout(
            title="Roczne przyrosty produkcji energii - S(t)",
            xaxis_title="Rok",
            yaxis_title="Przyrost roczny - S(t)",
            height=320
        )

        # =====================================================
        # INTERAKCJE
        # =====================================================

        interaction_wind = a_wind * F2_emp
        interaction_solar = a_solar * F1_emp

        fig_inter = go.Figure()

        fig_inter.add_trace(
            go.Scatter(
                x=data["Year"],
                y=interaction_wind,
                mode="lines+markers",
                name="a₁₂F₂(t)"
            )
        )

        fig_inter.add_trace(
            go.Scatter(
                x=data["Year"],
                y=interaction_solar,
                mode="lines+markers",
                name="a₂₁F₁(t)"
            )
        )

        fig_inter.update_layout(
            title="Efekty interakcji - aF(t)",
            xaxis_title="Rok",
            yaxis_title="Wpływ",
            height=320
        )

        # =====================================================
        # WYŚWIETLENIE 2
        # =====================================================

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                fig_S,
                use_container_width=True
            )

        with col4:
            st.plotly_chart(
                fig_inter,
                use_container_width=True
            )

        # =====================================================
        # HAZARD
        # =====================================================

        hazard_wind = (
            p_wind
            + q_wind * F1_emp
            + a_wind * F2_emp
        )

        hazard_solar = (
            p_solar
            + q_solar * F2_emp
            + a_solar * F1_emp
        )

        fig_h = go.Figure()

        fig_h.add_trace(
            go.Scatter(
                x=data["Year"],
                y=hazard_wind,
                mode="lines+markers",
                name="h₁(t) — Wind"
            )
        )

        fig_h.add_trace(
            go.Scatter(
                x=data["Year"],
                y=hazard_solar,
                mode="lines+markers",
                name="h₂(t) — Solar"
            )
        )

        fig_h.update_layout(
            title="Hazard adopcj - h(t)",
            xaxis_title="Rok",
            yaxis_title="Hazard",
            height=320
        )

        # =====================================================
        # UDZIAŁY RYNKOWE
        # =====================================================

        share_wind = (
            data["Wind"]
            / (data["Wind"] + data["Solar"])
        )

        share_solar = (
            data["Solar"]
            / (data["Wind"] + data["Solar"])
        )

        fig_share = go.Figure()

        fig_share.add_trace(
            go.Scatter(
                x=data["Year"],
                y=share_wind,
                mode="lines+markers",
                name="Udział Wind"
            )
        )

        fig_share.add_trace(
            go.Scatter(
                x=data["Year"],
                y=share_solar,
                mode="lines+markers",
                name="Udział Solar"
            )
        )

        fig_share.update_layout(
            title="Udziały rynkowe",
            xaxis_title="Rok",
            yaxis_title="Udział",
            height=320
        )

        # =====================================================
        # WYŚWIETLENIE 3
        # =====================================================

        col5, col6 = st.columns(2)

        with col5:
            st.plotly_chart(
                fig_h,
                use_container_width=True
            )

        with col6:
            st.plotly_chart(
                fig_share,
                use_container_width=True
            )

        # =====================================================
        # DANE VS MODEL
        # =====================================================

        fig_fit = go.Figure()

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Wind"],
                mode="markers",
                name="Wind — dane"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=wind_fit,
                mode="lines",
                name="Wind — model"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Solar"],
                mode="markers",
                name="Solar — dane"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=solar_fit,
                mode="lines",
                name="Solar — model"
            )
        )

        fig_fit.update_layout(
            title="Dane vs model",
            xaxis_title="Rok",
            yaxis_title="Produkcja energii",
            height=350
        )

        # =====================================================
        # RESZTY
        # =====================================================

        fig_res = go.Figure()

        fig_res.add_trace(
            go.Bar(
                x=data["Year"],
                y=residuals_wind,
                name="Reszty Wind"
            )
        )

        fig_res.add_trace(
            go.Bar(
                x=data["Year"],
                y=residuals_solar,
                name="Reszty Solar"
            )
        )

        fig_res.update_layout(
            title="Reszty modelu",
            xaxis_title="Rok",
            yaxis_title="Błąd",
            barmode="group",
            height=350
        )

        # =====================================================
        # WYŚWIETLENIE 4
        # =====================================================

        col7, col8 = st.columns(2)

        with col7:
            st.plotly_chart(
                fig_fit,
                use_container_width=True
            )

        with col8:
            st.plotly_chart(
                fig_res,
                use_container_width=True
            )
        # =====================================================
        ## =====================================================
    # SCENARIOS TAB
    # =====================================================

    
    with scenario_tab:

        st.subheader("Scenariusze parametrów")

        # =================================================
        # RESET
        # =================================================

        if st.button("🔄 Reset parametrów"):

            st.session_state["p1_change"] = 0
            st.session_state["q1_change"] = 0
            st.session_state["m1_change"] = 0
            st.session_state["a12_change"] = 0

            st.session_state["p2_change"] = 0
            st.session_state["q2_change"] = 0
            st.session_state["m2_change"] = 0
            st.session_state["a21_change"] = 0

        # =================================================
        # SUWAKI
        # =================================================

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:

            p1_change = st.slider(
                "Zmiana p Wind (%)",
                -90,
                200,
                0,
                key="p1_change"
            )

            q1_change = st.slider(
                "Zmiana q Wind (%)",
                -90,
                200,
                0,
                key="q1_change"
            )

        with col_s2:

            m1_change = st.slider(
                "Zmiana m Wind (%)",
                -90,
                200,
                0,
                key="m1_change"
            )

            a12_change = st.slider(
                "Zmiana a₁₂ (%)",
                -200,
                200,
                0,
                key="a12_change"
            )

        with col_s3:

            p2_change = st.slider(
                "Zmiana p Solar (%)",
                -90,
                200,
                0,
                key="p2_change"
            )

            q2_change = st.slider(
                "Zmiana q Solar (%)",
                -90,
                200,
                0,
                key="q2_change"
            )

        with col_s4:

            m2_change = st.slider(
                "Zmiana m Solar (%)",
                -90,
                200,
                0,
                key="m2_change"
            )

            a21_change = st.slider(
                "Zmiana a₂₁ (%)",
                -200,
                200,
                0,
                key="a21_change"
            )

        # =================================================
        # PARAMETRY SCENARIUSZA
        # =================================================

        p1_s = p_wind * (1 + p1_change / 100)
        q1_s = q_wind * (1 + q1_change / 100)
        m1_s = m_wind * (1 + m1_change / 100)
        a12_s = a_wind * (1 + a12_change / 100)

        p2_s = p_solar * (1 + p2_change / 100)
        q2_s = q_solar * (1 + q2_change / 100)
        m2_s = m_solar * (1 + m2_change / 100)
        a21_s = a_solar * (1 + a21_change / 100)

        # =================================================
        # MODEL SCENARIUSZOWY
        # =================================================

        wind_model_s = (
            p1_s * m1_s
            + (q1_s - p1_s) * data["Wind"]
            - (q1_s / m1_s) * data["Wind"]**2
            + a12_s * data["Solar"]
        )

        solar_model_s = (
            p2_s * m2_s
            + (q2_s - p2_s) * data["Solar"]
            - (q2_s / m2_s) * data["Solar"]**2
            + a21_s * data["Wind"]
        )

        # =================================================
        # REKONSTRUKCJA Y(t)
        # =================================================

        Y1_s = [data["Wind"].iloc[0]]
        Y2_s = [data["Solar"].iloc[0]]

        for i in range(1, len(data)):

            Y1_s.append(
                Y1_s[-1] + wind_model_s.iloc[i]
            )

            Y2_s.append(
                Y2_s[-1] + solar_model_s.iloc[i]
            )

        Y1_s = np.array(Y1_s)
        Y2_s = np.array(Y2_s)

        # =================================================
        # S(t)
        # =================================================

        S1_s = wind_model_s.fillna(0)
        S2_s = solar_model_s.fillna(0)

        # =================================================
        # F(t)
        # =================================================

        F1_s = Y1_s / m1_s
        F2_s = Y2_s / m2_s

        # =================================================
        # f(t)
        # =================================================

        f1_s = np.gradient(F1_s)
        f2_s = np.gradient(F2_s)

        # =================================================
        # h(t)
        # =================================================

        h1_s = (
            p1_s
            + q1_s * F1_s
            + a12_s * F2_s
        )

        h2_s = (
            p2_s
            + q2_s * F2_s
            + a21_s * F1_s
        )

        # =================================================
        # RESZTY
        # =================================================

        residuals_wind_s = (
            data["S_wind"] - S1_s
        ).fillna(0)

        residuals_solar_s = (
            data["S_solar"] - S2_s
        ).fillna(0)

        # =================================================
        # FUNKCJA WYKRESU
        # =================================================

        def create_figure(
            title,
            y1,
            y2,
            yaxis
        ):

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data["Year"],
                    y=y1,
                    mode="lines+markers",
                    name="Wind"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data["Year"],
                    y=y2,
                    mode="lines+markers",
                    name="Solar"
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Rok",
                yaxis_title=yaxis,
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            return fig

        # =================================================
        # GŁÓWNE WYKRESY
        # =================================================

        fig_Y = create_figure(
            "Y(t) — poziom adopcji",
            Y1_s,
            Y2_s,
            "Y(t)"
        )

        fig_f = create_figure(
            "f(t) — tempo adopcji",
            f1_s,
            f2_s,
            "f(t)"
        )

        fig_S = create_figure(
            "S(t) — przyrosty",
            S1_s,
            S2_s,
            "S(t)"
        )

        fig_h = create_figure(
            "h(t) — hazard adopcji",
            h1_s,
            h2_s,
            "h(t)"
        )

        # =================================================
        # DANE VS MODEL
        # =================================================

        fig_fit = go.Figure()

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Wind"],
                mode="markers",
                name="Wind — dane"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=Y1_s,
                mode="lines",
                name="Wind — model"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=data["Solar"],
                mode="markers",
                name="Solar — dane"
            )
        )

        fig_fit.add_trace(
            go.Scatter(
                x=data["Year"],
                y=Y2_s,
                mode="lines",
                name="Solar — model"
            )
        )

        fig_fit.update_layout(
            title="Dane vs model",
            xaxis_title="Rok",
            yaxis_title="Produkcja energii",
            height=350
        )

        # =================================================
        # UDZIAŁY RYNKOWE
        # =================================================

        share_wind_s = (
            Y1_s / (Y1_s + Y2_s + 1e-9)
        )

        share_solar_s = (
            Y2_s / (Y1_s + Y2_s + 1e-9)
        )

        fig_share = go.Figure()

        fig_share.add_trace(
            go.Scatter(
                x=data["Year"],
                y=share_wind_s,
                mode="lines+markers",
                name="Udział Wind"
            )
        )

        fig_share.add_trace(
            go.Scatter(
                x=data["Year"],
                y=share_solar_s,
                mode="lines+markers",
                name="Udział Solar"
            )
        )

        fig_share.update_layout(
            title="Udziały rynkowe",
            xaxis_title="Rok",
            yaxis_title="Udział",
            height=350
        )

        # =================================================
        # RESZTY
        # =================================================

        fig_residuals = go.Figure()

        fig_residuals.add_trace(
            go.Bar(
                x=data["Year"],
                y=residuals_wind_s,
                name="Reszty Wind"
            )
        )

        fig_residuals.add_trace(
            go.Bar(
                x=data["Year"],
                y=residuals_solar_s,
                name="Reszty Solar"
            )
        )

        fig_residuals.update_layout(
            title="Reszty modelu",
            xaxis_title="Rok",
            yaxis_title="Błąd",
            barmode="group",
            height=350
        )

        # =================================================
        # WYŚWIETLANIE
        # =================================================

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                fig_Y,
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                fig_f,
                use_container_width=True
            )

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                fig_S,
                use_container_width=True
            )

        with col4:
            st.plotly_chart(
                fig_h,
                use_container_width=True
            )

        
        # =================================================
        # DOLNY RZĄD — 3 WYKRESY
        # =================================================

        bottom_col1, bottom_col2, bottom_col3 = st.columns(3)

        # =================================================
        # DANE VS MODEL
        # =================================================

        with bottom_col1:

            st.plotly_chart(
                fig_fit,
                use_container_width=True,
                key="scenario_fit"
            )

        # =================================================
        # UDZIAŁY RYNKOWE
        # =================================================

        with bottom_col2:

            st.plotly_chart(
                fig_share,
                use_container_width=True,
                key="scenario_share"
            )

        # =================================================
        # RESZTY
        # =================================================

        with bottom_col3:

            st.plotly_chart(
                fig_residuals,
                use_container_width=True,
                key="scenario_residuals"
            )


