from A2 import label_encoding, one_hot_encoding
import pandas as pd


def load_market(path="data.xlsx", sheet="marketing_campaign"):
    return pd.read_excel(path, sheet)


def convert():
    data = load_market()
    print("Original shape:", data.shape)

    education = data["Education"].tolist()
    edu_map = label_encoding(education)
    data["Education"] = [edu_map[x] for x in education]

    marital = data["Marital_Status"].tolist()
    marital_map = one_hot_encoding(marital)

    for status in marital_map:
        col_name = "Marital_" + str(status)
        data[col_name] = [1 if x == status else 0 for x in marital]

    data = data.drop(columns=["Marital_Status"])

    if "Dt_Customer" in data.columns:
        data = data.drop(columns=["Dt_Customer"])

    print("After encoding shape:", data.shape)
    print("Education mapping:", edu_map)
    print("Marital one-hot mapping:", marital_map)
    print("Columns now:", list(data.columns))
    print("Feature dimensionality increased because one-hot adds one col per category.")
    return data


if __name__ == "__main__":
    convert()
