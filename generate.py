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


def join_extra_data():
    stocks_df = {}
    stocks_extra_df = {}

    print("Joining extra data to existing data")
    for fname in os.listdir("weighted_data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"weighted_data/{fname}")
            sym = fname[:-4]
            stocks_df[sym] = df.replace(np.nan, 0)

    for fname in os.listdir("extra_data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"extra_data/{fname}")
            sym = fname[:-10]
            stocks_extra_df[sym] = df.replace(np.nan, 0)

    for sym, df in stocks_df.items():
        extra_df = stocks_extra_df[sym]
        extra_df["Symbol"] = pd.Series([df.iloc[0]["Symbol"] for i in range(11)])
        extra_df["Sector"] = pd.Series([df.iloc[0]["Sector"] for i in range(11)])
        extra_df["Industry"] = pd.Series([df.iloc[0]["Industry"] for i in range(11)])
        added_df = pd.concat(
            [df, extra_df],
            ignore_index=True,
        )
        added_df.to_csv(f"complete_data/{sym}.csv", index=False)


def generate_master_table():
    stocks = pd.DataFrame()

    for fname in os.listdir("complete_data"):
        if fname.endswith(".csv") and fname != "master.csv":
            df = pd.read_csv(f"complete_data/{fname}")
            sym = fname[:-4]
            df = df.replace(np.nan, 0)
            stocks["Date"] = df["Date"]
            stocks[f"{sym}"] = df["Close"]

    stocks.to_csv("complete_data/master.csv", index=False)


def generate():
    stocks_df = {}
    stocks_by_sector = {}
    stocks_by_industry = {}

    print("Loading all data to dataframes")
    for fname in os.listdir("data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"data/{fname}")
            sym = fname[:-4]
            sector = df.iloc[0]["Sector"]
            industry = df.iloc[0]["Industry"]
            stocks_df[sym] = df.replace(np.nan, 0)

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
    sample_stock = stocks_df["VZ"]

    for k, s in stocks_df.items():
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
            stocks_df[k] = s

    # Found all industries/sectors of the interested stocks
    interested_sectors = []
    interested_industries = []

    print("Grabbing all interested sectors and industries")
    for s in interested_stocks:
        df = stocks_df[s]
        sector = df.iloc[0]["Sector"]
        industry = df.iloc[0]["Industry"]
        interested_sectors.append(sector)
        interested_industries.append(industry)

    interested_sectors = np.unique(interested_sectors)
    interested_industries = np.unique(interested_industries)

    # Get day change
    print("Calculating all daily change")
    for s in stocks_df.keys():
        df = stocks_df[s]
        movement = (df["Close"] - df["Open"]) / df["Open"]
        df["Day Change"] = movement.fillna(0)

    # Get sector day change
    print("Calculating all daily change for sector")
    days = len(stocks_df["VZ"].index)

    for sector in stocks_by_sector.keys():
        sector_vol = pd.Series(data=np.zeros(days))
        sector_total_vol = pd.Series(data=np.zeros(days))

        for s in stocks_by_sector[sector]:
            df = stocks_df[s]
            sector_total_vol = sector_total_vol + df["Volume"]
            sector_vol = sector_vol + (df["Day Change"] * df["Volume"])

        for s in stocks_by_sector[sector]:
            df = stocks_df[s]
            df["Weighted Sector Change"] = sector_vol / sector_total_vol

    # Get industry day change
    print("Calculating all daily change for industry")
    for industry in stocks_by_industry.keys():
        industry_vol = pd.Series(data=np.zeros(days))
        industry_total_vol = pd.Series(data=np.zeros(days))

        for s in stocks_by_industry[industry]:
            df = stocks_df[s]
            industry_total_vol = industry_total_vol + df["Volume"]
            industry_vol = industry_vol + (df["Day Change"] * df["Volume"])

        for s in stocks_by_industry[industry]:
            df = stocks_df[s]
            df["Weighted Industry Change"] = industry_vol / industry_total_vol

    # Export stocks
    print("Exporting as csv to ./weighted_data")
    export_stocks = np.copy(interested_stocks)

    for in_s in interested_stocks:
        sector = stocks_df[in_s].iloc[0]["Sector"]
        export_stocks = np.append(export_stocks, stocks_by_sector[sector], 0)
        industry = stocks_df[in_s].iloc[0]["Industry"]
        export_stocks = np.append(export_stocks, stocks_by_industry[industry], 0)

    export_stocks = np.unique(export_stocks)
    for s in export_stocks:
        stocks_df[s].to_csv(f"weighted_data/{s}.csv", index=False)


def generate_by_sector():
    stocks_df = {}
    stocks_by_sector = {}
    stocks_by_industry = {}

    print("Grouping data by sector")
    for fname in os.listdir("weighted_data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"weighted_data/{fname}")
            sym = fname[:-4]
            sector = df.iloc[0]["Sector"]
            industry = df.iloc[0]["Industry"]
            stocks_df[sym] = df.replace(np.nan, 0)

            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = [sym]
            else:
                stocks_by_sector[sector].append(sym)

            if industry not in stocks_by_industry:
                stocks_by_industry[industry] = [sym]
            else:
                stocks_by_industry[industry].append(sym)

    # Export stocks by sector as individual csv
    for k in stocks_by_sector.keys():
        sector_df = pd.DataFrame()

        for sym in stocks_by_sector[k]:
            df = stocks_df[sym]
            sector_df["Date"] = df["Date"]
            sector_df["Weighted Sector Change"] = df["Weighted Sector Change"]
            sector_df[f"{sym} Change"] = df["Day Change"]

        fname = k.replace(" ", "")
        sector_df.to_csv(f"weighted_data/by_sector/{fname}.csv", index=False)


def generate_by_industry():
    stocks_df = {}
    stocks_by_sector = {}
    stocks_by_industry = {}

    print("Grouping data by industry")
    for fname in os.listdir("weighted_data"):
        if fname.endswith(".csv"):
            df = pd.read_csv(f"weighted_data/{fname}")
            sym = fname[:-4]
            sector = df.iloc[0]["Sector"]
            industry = df.iloc[0]["Industry"]
            stocks_df[sym] = df.replace(np.nan, 0)

            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = [sym]
            else:
                stocks_by_sector[sector].append(sym)

            if industry not in stocks_by_industry:
                stocks_by_industry[industry] = [sym]
            else:
                stocks_by_industry[industry].append(sym)

    # Export stocks by industry as individual csv
    for k in stocks_by_industry.keys():
        industry_df = pd.DataFrame()

        for sym in stocks_by_industry[k]:
            df = stocks_df[sym]
            industry_df["Date"] = df["Date"]
            industry_df["Weighted Industry Change"] = df["Weighted Industry Change"]
            industry_df[f"{sym} Change"] = df["Day Change"]

        fname = k.replace(" ", "").replace("/", "_")
        industry_df.to_csv(f"weighted_data/by_industry/{fname}.csv", index=False)


if __name__ == "__main__":
    # generate()
    # generate_by_sector()
    # generate_by_industry()
    # join_extra_data()
    generate_master_table()
