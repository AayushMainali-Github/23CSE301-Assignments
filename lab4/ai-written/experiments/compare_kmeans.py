# Performance comparison: actual student A11 vs AI-assisted implementation.
from pathlib import Path
from time import perf_counter
import sys, pandas as pd, numpy as np
BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent
sys.path.insert(0, str(BASE))
import dmtoolkit as dm
ORIG=ROOT/"original"; sys.path.insert(0,str(ORIG))
from A11 import kmeans as student_kmeans
DATA=BASE/"data"/"data.xlsx"; RES=BASE/"outputs"/"results"; FIG=BASE/"outputs"/"figures"

def inertia(points,labels,centroids):
    x=np.asarray(points,float); c=np.asarray(centroids,float); lab=np.asarray(labels,int); return float(np.sum((x-c[lab])**2))
def bench(fn,points,k,runs=50):
    times=[]; result=None
    for _ in range(runs):
        t=perf_counter(); result=fn(points,k); times.append(perf_counter()-t)
    return result, times

def main():
    RES.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    df=pd.read_excel(DATA,sheet_name="marketing_campaign")[["Income","Recency","MntWines","NumWebPurchases"]].dropna(); points=df.to_numpy(float).tolist(); k=3
    (slab,scent),st=bench(student_kmeans,points,k)
    def aiwrap(p,k):
        r=dm.kmeans_ai(p,k); return r["labels"],r["centroids"],r["iterations"],r["inertia"]
    (alab,acent,ait,aiin),at=bench(aiwrap,points,k)
    sin=inertia(points,slab,scent)
    rows=[{"algorithm":"Student Lab03","avg_time_s":np.mean(st),"min_time_s":np.min(st),"std_time_s":np.std(st),"iterations":"not exposed","inertia":sin,"cluster_sizes":str([slab.count(i) for i in range(k)])},
          {"algorithm":"AI package","avg_time_s":np.mean(at),"min_time_s":np.min(at),"std_time_s":np.std(at),"iterations":ait,"inertia":aiin,"cluster_sizes":str([alab.count(i) for i in range(k)])}]
    out=pd.DataFrame(rows); out.to_csv(RES/"kmeans_comparison.csv",index=False)
    # timing figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); ax.bar(["Student Lab03","AI package"],[np.mean(st),np.mean(at)]); ax.set_ylabel("Average runtime (s)"); ax.set_title("K-means runtime comparison (50 runs)"); fig.savefig(FIG/"kmeans_runtime_comparison.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    print(out.to_string(index=False)); print("Speedup x",np.mean(st)/np.mean(at))
if __name__=="__main__": main()
