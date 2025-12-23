# DCA Simulator - Wersja Polska 🇵🇱

Profesjonalne narzędzia do ilościowej analizy i optymalizacji portfeli inwestycyjnych z pełnym interfejsem w języku polskim.

## 🎯 Opis

To jest polska wersja aplikacji DCA Simulator z pełnym tłumaczeniem interfejsu użytkownika oraz profesjonalną terminologią finansową i ekonometryczną.

## 🌟 Główne Funkcje

### 1. 💵 Symulator DCA (Dollar Cost Averaging)
- Symulacja strategii uśredniania kosztów zakupu
- Strategia "Kup na Spadku" z konfigurowalnymi progami
- Porównanie z benchmarkiem "Kup i Trzymaj"
- Wizualizacja wyników i szczegółowe metryki

### 2. 📈 Analiza Instrumentu Finansowego
Kompletna analiza pojedynczych aktywów:
- **Analiza Standardowa**:
  - Analiza rozkładu zwrotów
  - Testy pierwiastka jednostkowego (ADF, KPSS)
  - Value at Risk (VaR) z modelem GARCH
  - Prognoza Monte Carlo
  - Analiza techniczna (Bollinger Bands, MACD, RSI)
  - Prognozy: Prophet, ARIMA, Holt-Winters

- **ARIMA-GARCH Monte Carlo**:
  - Zaawansowane modelowanie ekonometryczne
  - Model ARIMA dla warunkowej średniej
  - Model GARCH dla warunkowej wariancji (heteroskedastyczność)
  - Symulacje Monte Carlo z rozkładami normalnym i t-Studenta
  - Testy stacjonarności i autokorelacji
  - Wykresy wachlarzowe prognoz

### 3. 🎯 Modele Portfelowe
- **CAPM**: Analiza współczynnika beta i oczekiwanych zwrotów
- **Optymalizacja Monte Carlo**: Maksymalizacja współczynnika Sharpe'a
- **Zaawansowana analiza portfela**:
  - HRP (Hierarchical Risk Parity)
  - Mean-Variance (Markowitz)
  - Minimum Variance
  - Risk Parity
  - Equal Weight
  - Inverse Volatility
- Testy stresowe i analiza scenariuszowa
- Prognoza Monte Carlo dla portfela
- Tryby: Backtesting vs Production

### 4. 🔬 Backtest Portfela
- Testowanie dowolnego portfela z niestandardowymi wagami
- Porównanie z benchmarkiem
- Szczegółowe metryki wydajności
- Eksport wyników do CSV

## 📋 Wymagania

```bash
streamlit
pandas
numpy
yfinance
plotly
scipy
statsmodels
arch
prophet
scikit-learn
```

## 🚀 Instalacja i Uruchomienie

1. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

2. Uruchom aplikację:
```bash
streamlit run app_new.py
```

## 🎓 Profesjonalna Terminologia

Aplikacja wykorzystuje precyzyjną terminologię ekonometryczną i finansową:

### Ekonometria:
- **Stacjonarność** - właściwość szeregu czasowego o stałych parametrach
- **Heteroskedastyczność** - zmienna wariancja reszt modelu
- **Leptokurtoza** - grube ogony rozkładu (excess kurtosis)
- **Innowacje** - nieprzewidywalne składniki losowe w szeregu
- **Test ADF** - test Dickeya-Fullera stacjonarności
- **Test KPSS** - test stacjonarności wokół trendu
- **ACF/PACF** - funkcje autokorelacji

### Finanse:
- **Współczynnik Sharpe'a** - stosunek nadwyżkowego zwrotu do ryzyka
- **VaR (Value at Risk)** - wartość narażona na ryzyko
- **Drawdown** - maksymalny spadek wartości portfela
- **Współczynnik Beta** - wrażliwość aktywa na ruchy rynku
- **CAPM** - model wyceny aktywów kapitałowych

### Modelowanie:
- **ARIMA(p,d,q)** - autoregresyjny zintegrowany model średniej ruchomej
- **GARCH(p,q)** - uogólniony autoregresyjny model heteroskedastyczności warunkowej
- **Monte Carlo** - metoda symulacji stochastycznej
- **Prognoza wachlarzowa** - przedziały ufności prognoz

## 📊 Struktura Projektu

```
dca_simulator_pl/
├── app_new.py                 # Główna aplikacja (URUCHOM TEN PLIK)
├── config.py                  # Konfiguracja
├── data_fetcher.py           # Pobieranie danych z Yahoo Finance
├── dca_engine.py             # Silnik symulacji DCA
├── visualizer.py             # Wizualizacje
├── requirements.txt          # Zależności
├── pages/
│   ├── dca_page.py                    # Symulator DCA
│   ├── financial_analysis_page.py     # Analiza standardowa
│   ├── arima_garch_page.py           # ARIMA-GARCH Monte Carlo
│   ├── portfolio_models_page.py      # Modele portfelowe
│   └── portfolio_backtest_page.py    # Backtest portfela
├── portfolio/
│   ├── hrp.py                # Hierarchical Risk Parity
│   ├── optimization.py       # Optymalizacja portfela
│   └── monte_carlo.py       # Symulacje Monte Carlo
└── analysis/
    └── (moduły analityczne)
```

## 🔄 Różnice od wersji angielskiej

- ✅ Pełne tłumaczenie wszystkich elementów interfejsu
- ✅ Profesjonalna terminologia ekonometryczna i finansowa
- ✅ Wszystkie nagłówki, przyciski, komunikaty po polsku
- ✅ Tłumaczenia metodologii i założeń
- ✅ Polskie komunikaty błędów i ostrzeżeń
- ✅ Zachowanie pełnej funkcjonalności oryginalnej wersji

## ⚠️ Ważne Uwagi

- Aplikacja wykorzystuje dane historyczne - przeszłe wyniki nie gwarantują przyszłych rezultatów
- Backtesting nie uwzględnia kosztów transakcyjnych, podatków ani poślizgu cenowego
- Modele zakładają pewne uproszczenia (np. płynność rynku, brak spread'u bid-ask)
- Do celów edukacyjnych i analitycznych - nie stanowi porady inwestycyjnej

## 📝 Changelog

### Wersja Polska 1.0 (23.12.2025)
- Pełne tłumaczenie interfejsu użytkownika na język polski
- Implementacja modułu ARIMA-GARCH Monte Carlo
- Dodanie profesjonalnej terminologii ekonometrycznej
- Tłumaczenie wszystkich komunikatów, przycisków i etykiet
- Aktualizacja dokumentacji

## 👨‍💻 Autor

Wersja polska: Tłumaczenie i adaptacja 2025

Oryginalna aplikacja: [quant-lab](https://github.com/MikPe1/quant-lab)

## 📄 Licencja

Zgodnie z licencją projektu oryginalnego.

---

**Quant Lab PL** - Profesjonalne narzędzia analizy ilościowej po polsku 🇵🇱
