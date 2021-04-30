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
            stocks_pd[sym] = df.replace(np.nan, 0)

            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = [sym]
            else:
                stocks_by_sector[sector].append(sym)

            if industry not in stocks_by_industry:
                stocks_by_industry[industry] = [sym]
            else:
                stocks_by_industry[industry].append(sym)

    # Fix dates
    replacements = {}
    sample_stock = stocks_pd["VZ"]

    for k, s in stocks_pd.items():
        idx = sample_stock[
            sample_stock["Date"] == s.iloc[0]["Date"]
        ].index.values.astype(int)[0]
        if idx > 0:
            empty_rows_arr = [
                pd.DataFrame(
                    [
                        [
                            sample_stock.iloc[i]["Date"],
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            s.iloc[0]["Symbol"],
                            s.iloc[0]["Sector"],
                            s.iloc[0]["Industry"],
                        ]
                    ],
                    columns=list(s.columns),
                )
                for i in range(idx)
            ]
            empty_rows = pd.concat(empty_rows_arr, ignore_index=True)
            empty_rows = empty_rows.replace(np.nan, 0)
            replacements[k] = pd.concat([empty_rows, s], ignore_index=True)

        for k, s in replacements.items():
            stocks_pd[k] = s

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
    for s in stocks_pd.keys():
        df = stocks_pd[s]
        movement = (df["Close"] - df["Open"]) / df["Open"]
        df["Day Change"] = movement.fillna(0)

    # Get sector day change
    print("Calculating all daily change for sector")
    days = len(stocks_pd["VZ"].index)

    for sector in stocks_by_sector.keys():
        sector_vol = pd.Series(data=np.zeros(days))
        sector_total_vol = pd.Series(data=np.zeros(days))

        for s in stocks_by_sector[sector]:
            df = stocks_pd[s]
            sector_total_vol = sector_total_vol + df["Volume"]
            sector_vol = sector_vol + (df["Day Change"] * df["Volume"])

        for s in stocks_by_sector[sector]:
            df = stocks_pd[s]
            df["Weighted Sector Change"] = sector_vol / sector_total_vol

    # Get industry day change
    print("Calculating all daily change for industry")
    for industry in stocks_by_industry.keys():
        industry_vol = pd.Series(data=np.zeros(days))
        industry_total_vol = pd.Series(data=np.zeros(days))

        for s in stocks_by_industry[industry]:
            df = stocks_pd[s]
            industry_total_vol = industry_total_vol + df["Volume"]
            industry_vol = industry_vol + (df["Day Change"] * df["Volume"])

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
        stocks_pd[s].to_csv(f"weighted_data/{s}.csv", index=False)


if __name__ == "__main__":
    generate()
