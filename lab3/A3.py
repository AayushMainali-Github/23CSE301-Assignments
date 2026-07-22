from A2 import label_encoding, one_hot_encoding
import pandas as pd

def load_market(path,sheet):
    return pd.read_excel(path, sheet)

def convert():
    data = load_market("data.xlsx", "market_campaign")

    education = data['Education']
    marital_status = data['Marita_Status']

    label_education = label_encoding(education)
    one_hot_marital = one_hot_encoding(marital_status)