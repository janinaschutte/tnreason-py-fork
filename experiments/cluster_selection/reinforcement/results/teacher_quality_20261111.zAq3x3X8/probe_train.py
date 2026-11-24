"""Read-only post-hoc greedy construction on the 24 frozen training contexts."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.cluster_selection.reinforcement import learn
from experiments.cluster_selection.reinforcement.model import load
from experiments.cluster_selection.reinforcement.train import _closed, _state


def main():
    output=Path(__file__).resolve().parent
    if not (output/"summary.json").exists():
        raise ValueError("Run this read-only probe after all controlled solves finish")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol=json.loads((output/"protocol.json").read_text())
    old=ROOT/"experiments/cluster_selection/reinforcement/results/constructive_improvement_20261106.903639e9/states.json"
    records=[r for r in json.loads(old.read_text()) if r["split"]=="train"]
    data_path=ROOT/"experiments/cluster_selection/reinforcement/results/teacher_forks_scaled_20261028.q9m8bpg9/development.json"
    data=json.loads(data_path.read_text())
    if sha256(old.read_bytes()).hexdigest()!=protocol["states_sha256"] or sha256(data_path.read_bytes()).hexdigest()!=protocol["data_sha256"]:
        raise AssertionError("Frozen data or states changed")
    state=_state({**data,"samples":[dict(domains=[list(range(d)) for d in data["network"]["dimensions"]])]})
    checkpoints={"parent":ROOT/protocol["parent"],**{arm:output/f"{arm}.pt" for arm in protocol["controls"]["arms"]}}
    rows=[]
    for name,path in checkpoints.items():
        model=load(path).cpu().eval()
        model.beta=0.
        with torch.no_grad():
            for record in records:
                with _closed(state,record,model.network["colors"]):
                    before=state.snapshot_domains()
                    candidate,_,_,trace=model.rollout(state.domains,greedy=True,beta=0.)
                    metrics=learn.measure(model,state,candidate,objective="pruning",known_satisfiable=record["known_satisfiable"])
                    if before!=state.snapshot_domains():
                        raise AssertionError("Read-only trial changed frozen domains")
                    rows.append(dict(model=name,state=record["id"],puzzle=record["puzzle"],root=record["root"],
                        known_satisfiable=record["known_satisfiable"],steps=len(trace),metrics=metrics))
        selected=[r for r in rows if r["model"]==name]
        print("train greedy",name,sum(r["metrics"]["useful"] for r in selected),"/",len(selected),flush=True)
    groups={}
    for name in checkpoints:
        groups[name]={}
        for split in ("all","root","branch"):
            selected=[r for r in rows if r["model"]==name and (split=="all" or r["root"]==(split=="root"))]
            groups[name][split]=dict(states=len(selected),useful=sum(r["metrics"]["useful"] for r in selected),
                mean_reward=sum(r["metrics"]["reward"] for r in selected)/max(1,len(selected)),
                solved=sum(r["metrics"]["solved"] for r in selected),contradictions=sum(r["metrics"]["contradictions"] for r in selected))
    result=dict(version=1,read_only=True,no_retraining=True,no_selection=True,states=24,groups=groups,rows=rows,
        protocol_sha256=sha256((output/"protocol.json").read_bytes()).hexdigest(),
        states_sha256=sha256(old.read_bytes()).hexdigest(),
        checkpoint_sha256={name:sha256(path.read_bytes()).hexdigest() for name,path in checkpoints.items()},
        code_sha256={str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),Path(learn.__file__),Path(learn.__file__).with_name("model.py"))},
        note="Unforced one-cluster greedy construction, distinct from discrepancy DFS. Training contains root and DFS contexts, including unknown satisfiability branches; validation contains satisfiable roots.")
    with (output/"train_greedy_probe.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__=="__main__":
    main()
