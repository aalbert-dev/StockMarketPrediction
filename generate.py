import os
import pandas as pd
import numpy as np

interested_stocks = [
    "VZ",
    "T",
    "WMT",
    "MGM",
    "GPS",
    "GT",
    "BBY",
    "AFG",
    "ERJ",
    "MYE",
    "ECPG",
    "GCO",
    "MPC",
    "TRI",
    "UFI",
]


def generate():
    stocks_pd = {}
    stocks_by_sector = {}
    stocks_by_industry = {}

    print("Loading all data to dataframes")
    for fname in os.listdir("data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"data/{fname}")
            sym = fname[:-4]
            sector = df.iloc[0]["Sector"]
            industry = df.iloc[0]["Industry"]
            stocks_pd[sym] = df

            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = [sym]
            else:
                stocks_by_sector[sector].append(sym)

            if industry not in stocks_by_industry:
                stocks_by_industry[industry] = [sym]
            else:
                stocks_by_industry[industry].append(sym)

    # Found all industries/sectors of the interested stocks
    interested_sectors = []
    interested_industries = []

    print("Grabbing all interested sectors and industries")
    for s in interested_stocks:
        df = stocks_pd[s]
        sector = df.iloc[0]["Sector"]
        industry = df.iloc[0]["Industry"]
        interested_sectors.append(sector)
        interested_industries.append(industry)

    interested_sectors = np.unique(interested_sectors)
    interested_industries = np.unique(interested_industries)

    # Get day change
    print("Calculating all daily change")
    for s in interested_stocks:
        df = stocks_pd[s]
        movement = (df["Close"] - df["Open"]) / df["Open"]
        df["Day Change"] = movement

    # Get sector day change
    print("Calculating all daily change for sector")
    for in_s in interested_stocks:
        sector = stocks_pd[in_s].iloc[0]["Sector"]
        days = len(stocks_pd[in_s].index)
        sector_vol = pd.Series(data=np.zeros(days))
        sector_total_vol = pd.Series(data=np.zeros(days))
        for s in stocks_by_sector[sector]:
            df = stocks_pd[s]
            movement = (df["Close"] - df["Open"]) / df["Open"]
            df["Day Change"] = movement
            sector_total_vol = sector_total_vol + df["Volume"]
            sector_vol = sector_vol + (movement * df["Volume"])

        for s in stocks_by_sector[sector]:
            df = stocks_pd[s]
            df["Weighted Sector Change"] = sector_vol / sector_total_vol

    # Get industry day change
    print("Calculating all daily change for industry")
    for in_s in interested_stocks:
        industry = stocks_pd[in_s].iloc[0]["Industry"]
        days = len(stocks_pd[in_s].index)
        industry_vol = pd.Series(data=np.zeros(days))
        industry_total_vol = pd.Series(data=np.zeros(days))

        for s in stocks_by_industry[industry]:
            df = stocks_pd[s]
            movement = (df["Close"] - df["Open"]) / df["Open"]
            df["Day Change"] = movement
            industry_total_vol = industry_total_vol + df["Volume"]
            industry_vol = industry_vol + (movement * df["Volume"])
            print(f"total: {industry_total_vol.head()}")
            print(f"industry: {industry_vol.head()}")

        for s in stocks_by_industry[industry]:
            df = stocks_pd[s]
            df["Weighted Industry Change"] = industry_vol / industry_total_vol

    # Export stocks
    print("Exporting as csv to ./weighted_data")
    export_stocks = np.copy(interested_stocks)

    for in_s in interested_stocks:
        sector = stocks_pd[in_s].iloc[0]["Sector"]
        export_stocks = np.append(export_stocks, stocks_by_sector[sector], 0)
        industry = stocks_pd[in_s].iloc[0]["Industry"]
        export_stocks = np.append(export_stocks, stocks_by_industry[industry], 0)

    export_stocks = np.unique(export_stocks)
    for s in export_stocks:
        stocks_pd[s].to_csv(f"weighted_data/{s}.csv")


if __name__ == "__main__":
    generate()
