# GenAI-assisted unit tests: ChatGPT (GPT-5.6 Sol)
import pandas as pd
from dmtoolkit.preprocessing import label_encode, one_hot_encode, encode_dataframe

def test_label_encode():
    encoded, mapping = label_encode(["A","B","A","C"])
    assert encoded == [0,1,0,2] and mapping == {"A":0,"B":1,"C":2}

def test_one_hot_encode():
    encoded, mapping = one_hot_encode(["Red","Blue","Red"], prefix="Color")
    assert encoded.shape == (3,2)
    assert mapping["Red"] == [1,0] and mapping["Blue"] == [0,1]

def test_encode_dataframe():
    df = pd.DataFrame({"Education":["G","P","G"], "Status":["S","M","S"], "Income":[1,2,3]})
    out, mappings = encode_dataframe(df, ["Education"], ["Status"])
    assert "Status" not in out.columns and "Status_S" in out.columns and "Status_M" in out.columns
    assert mappings["Education"]["type"] == "label"

def test_empty_label_input():
    import pytest
    with pytest.raises(ValueError): label_encode([])
