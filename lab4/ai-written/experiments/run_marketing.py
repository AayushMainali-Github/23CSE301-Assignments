# GenAI-assisted Lab03 repeat: ChatGPT (GPT-5.6 Sol)
from pathlib import Path
import sys
import pandas as pd, numpy as np
from scipy.spatial.distance import minkowski
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import dmtoolkit as dm
DATA=BASE/"data"/"data.xlsx"; FIG=BASE/"outputs"/"figures"; RES=BASE/"outputs"/"results"

def main():
    FIG.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
    raw=pd.read_excel(DATA,sheet_name="marketing_campaign")
    enc,mappings=dm.encode_dataframe(raw,["Education"],["Marital_Status"],["Dt_Customer"])
    numeric=enc.select_dtypes(include="number").dropna()
    v1,v2=numeric.iloc[0].tolist(),numeric.iloc[1].tolist()
    ps=list(range(1,11)); ds=[]; rows=[]
    for p in ps:
        own=dm.minkowski_distance(v1,v2,p); ref=minkowski(v1,v2,p); ds.append(own); rows.append({"p":p,"custom":own,"scipy":ref,"abs_error":abs(own-ref)})
    dm.plot_minkowski(ps,ds,FIG/"ai_minkowski.png")
    pd.DataFrame(rows).to_csv(RES/"distance_comparison.csv",index=False)
    matrix=numeric.to_numpy(float); means,vars_,stds=dm.dataset_statistics(matrix.tolist())
    pd.DataFrame({"feature":numeric.columns,"custom_mean":means,"numpy_mean":matrix.mean(0),"custom_variance":vars_,"numpy_variance":matrix.var(0),"custom_std":stds,"numpy_std":matrix.std(0)}).to_csv(RES/"statistics_comparison.csv",index=False)
    income=numeric["Income"].tolist(); dm.plot_histogram(income,20,"Income",FIG/"ai_income_histogram.png")
    cols=["Income","Recency","MntWines","NumWebPurchases"]; points=numeric[cols].to_numpy(float).tolist(); result=dm.kmeans_ai(points,3)
    dm.plot_clusters(points,result["labels"],result["centroids"],(cols[0],cols[1]),FIG/"ai_kmeans_clusters.png")
    pd.DataFrame({"metric":["rows_after_dropna","features_after_encoding","income_mean","income_variance","kmeans_iterations","kmeans_inertia"],"value":[len(numeric),numeric.shape[1],dm.mean(income),dm.variance(income),result["iterations"],result["inertia"]]}).to_csv(RES/"ai_summary.csv",index=False)
    print("AI experiments complete")
    print("Original shape:", raw.shape, "Encoded numeric shape:", numeric.shape)
    print("K-means iterations:", result["iterations"], "inertia:", result["inertia"], "cluster sizes:", [result["labels"].count(i) for i in range(3)])
if __name__=="__main__": main()
