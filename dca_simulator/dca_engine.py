import pandas as pd
from datetime import timedelta


def simulate_dca(
    historical_data: pd.DataFrame,
    initial_capital: float,
    regular_investment_amount: float,
    investment_interval: str,
    buy_on_dip_threshold: float = 0,
    buy_on_dip_multiplier: float = 1
) -> pd.DataFrame:

    if historical_data.empty:
        return pd.DataFrame()

    results = []
    total_shares = 0.0
    total_invested = 0.0
    last_investment_date = None

    interval_days = {
        "Daily": 1,
        "Weekly": 7,
        "Bi-Weekly": 14,
        "Monthly": 30
    }
    interval = timedelta(days=interval_days.get(investment_interval, 30))

    previous_price = None
    for index, value in historical_data.items():
        current_date = index
        current_price = float(value)

        shares_bought_today = 0.0
        dip_triggered = False

        # Check for 'buy on dip' condition based on daily return
        if previous_price is not None and buy_on_dip_threshold > 0:
            daily_return = ((previous_price - current_price) / previous_price) * 100
            if daily_return <= -buy_on_dip_threshold:  # Negative return >= threshold
                dip_triggered = True

        if total_invested == 0 and initial_capital > 0:
            shares_bought_today = initial_capital / current_price
            total_shares += shares_bought_today
            total_invested += initial_capital
            last_investment_date = current_date
        else:
            if last_investment_date is None or (current_date - last_investment_date) >= interval:
                amount_to_invest = regular_investment_amount

                if dip_triggered:
                    amount_to_invest *= buy_on_dip_multiplier

                if amount_to_invest > 0:
                    shares_bought_today = amount_to_invest / current_price
                    total_shares += shares_bought_today
                    total_invested += amount_to_invest
                    last_investment_date = current_date

        previous_price = current_price

        portfolio_value = total_shares * current_price
        profit_loss = portfolio_value - total_invested

        results.append({
            "Date": current_date,
            "Price": float(current_price),
            "Shares Bought": float(shares_bought_today),
            "Total Shares": float(total_shares),
            "Total Invested": float(total_invested),
            "Portfolio Value": float(portfolio_value),
            "Profit/Loss": float(profit_loss),
            "Dip Triggered": dip_triggered
        })

    # Calculate buy-and-hold
    first_price = historical_data.iloc[0]
    shares_bh = total_invested / first_price
    for result in results:
        bh_value = shares_bh * result['Price']
        result['Buy&Hold Value'] = float(bh_value)
        result['Buy&Hold Profit/Loss'] = float(bh_value - total_invested)

    return pd.DataFrame(results)

