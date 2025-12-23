"""
Strona Symulatora DCA - Symulacja i wizualizacja uśredniania kosztów zakupu.
"""

import streamlit as st
import datetime
import config
from data_fetcher import fetch_historical_data
from dca_engine import simulate_dca
from visualizer import plot_simulation_results, display_results_table


def render_dca_page():
    """Renderowanie strony Symulatora DCA."""
    
    st.title("💵 Symulator DCA (Uśrednianie Kosztów Zakupu)")
    with st.expander("📚 Założenia i Metodologia", expanded=False):
        st.markdown("""
        **Dollar Cost Averaging (DCA)** to strategia inwestycyjna polegająca na regularnym inwestowaniu 
        stałej kwoty w ustalone okresy, niezależnie od bieżącej ceny aktywa. Symulator umożliwia porównanie 
        standardowej strategii DCA z opcjonalną modyfikacją **"Kup na Spadku"**, która zwiększa inwestycję 
        po znaczących spadkach ceny.
        
        **Kluczowe założenia:**
        - Wszystkie inwestycje realizowane są po cenie zamknięcia (brak transakcji śródsesyjnych)
        - Nie uwzględnia kosztów transakcyjnych, podatków ani poślizgu cenowego
        - Zakłada pełną płynność rynku dla wszystkich transakcji
        - Wykorzystuje dane historyczne; przeszłe wyniki nie gwarantują przyszłych rezultatów
        - "Kup na Spadku" aktywuje się przy spadku ceny o określony procent względem poprzedniego okresu
        - Benchmark "Kup i Trzymaj" zakłada jednorazową inwestycję całego kapitału na starcie
        
        **Metodologia:**
        - W każdym interwale inwestowana jest określona kwota
        - Jeśli "Kup na Spadku" jest aktywny i próg spadku został osiągnięty, kwota inwestycji jest mnożona
        - Śledzenie wartości portfela, całkowitej zainwestowanej kwoty oraz zysku/straty w czasie
        - Porównanie wyników ze strategią "Kup i Trzymaj" dla tego samego okresu
        """)
    
    st.sidebar.header("⚙️ Parametry Inwestycyjne")

    with st.sidebar.form("investment_form"):
        ticker = st.text_input(
            "Symbol giełdowy (np. AAPL)", 
            config.DEFAULT_TICKER
        ).upper()
        
        start_date = st.date_input(
            "Data początkowa", 
            datetime.datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date()
        )
        
        initial_capital = st.number_input(
            "Kapitał początkowy ($)", 
            min_value=0, 
            value=config.DEFAULT_INITIAL_CAPITAL, 
            step=100
        )
        
        regular_investment_amount = st.number_input(
            "Regularna kwota inwestycji ($)", 
            min_value=0, 
            value=config.DEFAULT_REGULAR_INVESTMENT_AMOUNT, 
            step=10
        )
        
        investment_interval = st.selectbox(
            "Interwał inwestycyjny", 
            ["Dzienny", "Tygodniowy", "Dwutygodniowy", "Miesięczny"], 
            index=["Daily", "Weekly", "Bi-Weekly", "Monthly"].index(config.DEFAULT_INVESTMENT_INTERVAL)
        )

        st.sidebar.header("📈 Strategia Zaawansowana: Kup na Spadku")
        enable_buy_on_dip = st.checkbox("Włącz strategię 'Kup na Spadku'", value=False)
        
        buy_on_dip_threshold = st.slider(
            "Próg spadku (%)", 
            min_value=1, 
            max_value=20, 
            value=config.DEFAULT_BUY_ON_DIP_THRESHOLD, 
            step=1, 
            disabled=not enable_buy_on_dip
        )
        
        buy_on_dip_multiplier = st.slider(
            "Mnożnik inwestycji przy spadku", 
            min_value=1.0, 
            max_value=5.0, 
            value=config.DEFAULT_BUY_ON_DIP_MULTIPLIER, 
            step=0.5, 
            disabled=not enable_buy_on_dip
        )

        submitted = st.form_submit_button("Uruchom Symulację")
    
    if submitted:
        st.write(f"Uruchamianie symulacji dla **{ticker}** od **{start_date}**")
        interval_map = {"Dzienny": "Daily", "Tygodniowy": "Weekly", "Dwutygodniowy": "Bi-Weekly", "Miesięczny": "Monthly"}
        investment_interval_en = interval_map.get(investment_interval, "Daily")
        st.write(
            f"Kapitał początkowy: ${initial_capital:,.2f}  |  Regularna inwestycja: ${regular_investment_amount:,.2f} ({investment_interval})"
        )
        
        if enable_buy_on_dip:
            st.info(f"'Kup na Spadku' włączony z progiem: {buy_on_dip_threshold}% i mnożnikiem: {buy_on_dip_multiplier}x")
        else:
            st.info("Strategia 'Kup na Spadku' jest wyłączona.")
    
        # Fetch historical data
        start_date_str = start_date.strftime("%Y-%m-%d")
        historical_data = fetch_historical_data(ticker, start_date_str)
        
        if historical_data.empty:
            st.error(f"Nie znaleziono danych historycznych dla '{ticker}' od {start_date_str}. Sprawdź symbol i datę.")
        else:
            # Run simulation
            buy_on_dip_thresh = buy_on_dip_threshold if enable_buy_on_dip else 0
            buy_on_dip_mult = buy_on_dip_multiplier if enable_buy_on_dip else 1.0
            
            with st.spinner("Uruchamianie symulacji DCA..."):
                simulation_results = simulate_dca(
                    historical_data,
                    initial_capital,
                    regular_investment_amount,
                    investment_interval,
                    buy_on_dip_thresh,
                    buy_on_dip_mult
                )
            
            if simulation_results.empty:
                st.error("Symulacja nie powiodła się. Sprawdź parametry.")
            else:
                # Display results
                st.success("✅ Symulacja zakończona!")
                st.subheader("Wyniki Symulacji")
                
                # Summary metrics
                final_row = simulation_results.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Końcowa Wartość Portfela", 
                        f"${final_row['Portfolio Value']:,.2f}"
                    )
                with col2:
                    st.metric(
                        "Całkowita Inwestycja", 
                        f"${final_row['Total Invested']:,.2f}"
                    )
                with col3:
                    profit_loss = final_row['Profit/Loss']
                    st.metric(
                        "Zysk/Strata", 
                        f"${profit_loss:,.2f}",
                        delta=f"{(profit_loss / final_row['Total Invested'] * 100):.1f}%"
                    )
                with col4:
                    bh_profit = final_row['Buy&Hold Profit/Loss']
                    st.metric(
                        "Zysk/Strata Kup i Trzymaj", 
                        f"${bh_profit:,.2f}",
                        delta=f"{(bh_profit / final_row['Total Invested'] * 100):.1f}%"
                    )
                
                # Plot
                fig = plot_simulation_results(simulation_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                with st.expander("📊 Pokaż Szczegółową Tabelę Wyników"):
                    results_table = display_results_table(simulation_results)
                    st.dataframe(results_table, use_container_width=True)
                    
                    # Download button
                    csv = simulation_results.to_csv(index=False)
                    st.download_button(
                        label="Pobierz Wyniki jako CSV",
                        data=csv,
                        file_name=f"dca_simulation_{ticker}_{start_date_str}.csv",
                        mime="text/csv"
                    )
