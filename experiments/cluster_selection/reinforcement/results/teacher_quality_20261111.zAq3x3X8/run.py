"""Frozen-state, equal-update diagnostic of teacher quality and imitation loss."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.cluster_selection.candidate_generation import ClusterCandidateGenerator
from experiments.cluster_selection.reinforcement import learn
from experiments.cluster_selection.reinforcement.diagnose import teachers_for
from experiments.cluster_selection.reinforcement.experiment import fingerprints, instance_hashes, instances, solve
from experiments.cluster_selection.reinforcement.model import load
from experiments.cluster_selection.reinforcement.train import _closed, _save, _state

OUT = Path(__file__).resolve().parent
OLD = ROOT / "experiments/cluster_selection/reinforcement/results/constructive_improvement_20261106.903639e9"
DATA = ROOT / "experiments/cluster_selection/reinforcement/results/teacher_forks_scaled_20261028.q9m8bpg9/development.json"
PARENT = DATA.parent / "warmup/policy.best.pt"
ARMS = {"direct64_uniformmass": ("direct",0.,False), "nested256_uniformmass": ("nested",0.,False),
        "nested256_nearbest_mass": ("nested",.9,False), "nested256_nearbest_greedy": ("nested",.9,True)}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,separators=(",",":"),allow_nan=False)
        stream.write("\n")


def distribution(values):
    return dict(n=len(values),mean=float(np.mean(values)),min=float(min(values)),max=float(max(values)),
                quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist()) if values else dict(n=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("prepare","train","all"),default="all")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(29)
    data=json.loads(DATA.read_text())
    old=json.loads((OLD/"protocol.json").read_text())
    frozen=json.loads((OLD/"states.json").read_text())
    training=[{k:v for k,v in r.items() if k not in ("teachers","tested")} for r in frozen if r["split"]=="train"]
    validation=[r for r in frozen if r["split"]=="validation"]
    ids=old["validation_puzzles"]
    initial,_=instances(data)
    if digest(DATA)!=old["data_sha256"] or len(training)!=24 or len(validation)!=15 or len(ids)!=23:
        raise AssertionError("Expected the frozen 24/15 states and 23 clean validation puzzles")
    if set(ids)&set(data["train_indices"]) or {r["puzzle"] for r in training}-set(data["train_indices"]):
        raise AssertionError("Training/validation index overlap")
    if fingerprints(data,[initial[i] for i in ids])&fingerprints(data,[initial[i] for i in data["train_indices"]]):
        raise AssertionError("Training/validation initial-domain overlap")
    saved=torch.load(PARENT,map_location="cpu",weights_only=True)
    inherited=saved.get("training",{}).get("train_instance_hashes")
    if inherited is None or set(inherited)&instance_hashes(data,[initial[i] for i in ids]):
        raise AssertionError("Parent has unknown or overlapping validation exposure")
    parent=load(PARENT).cpu().eval()
    if parent.network!=data["network"] or not parent.config.get("target_first"):
        raise AssertionError("Expected the same target-first network")
    parent.beta=0.
    parent.config.pop("max_total_work",None)
    parent.config["decoder"]="discrepancy"
    state=_state({**data,"samples":[dict(domains=[list(range(d)) for d in data["network"]["dimensions"]])]})
    paths=[Path(__file__),Path(learn.__file__),Path(learn.__file__).with_name("model.py"),
           Path(learn.__file__).with_name("train.py"),Path(learn.__file__).with_name("experiment.py"),
           ROOT/"experiments/cluster_selection/runtime.py",ROOT/"experiments/cluster_selection/candidate_generation.py",
           ROOT/"experiments/cluster_selection/candidate_evaluation_mac.py",ROOT/"experiments/constraint_networks/mac.py"]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    arm_order=list(ARMS)
    Random(30).shuffle(arm_order)
    controls=dict(seed=29,lr=.0003,updates=240,passes=10,train_states=24,validation_states=15,
        validation_puzzles=ids,arm_order=arm_order,arms=ARMS,probe_limit=80,mutations=32,
        probe_seed="29+stateid, same causal proposals for both pools",max_cores=64,max_open_colors=6,
        max_work=1_000_000,max_size=100_000,compact_domains=True,decoder="discrepancy",
        cluster_allowance=32,decoder_steps=8192,node_guard=10000,seconds_guard=120,total_work=None,
        selection="Final model after exactly 240 updates; no validation checkpoint selection",
        objective="Measured pruning reward; not DFS node supervision",no_test_queries=True,
        excluded_old_validation=old["excluded_old_validation_puzzles"])
    protocol=dict(version=1,kind="controlled_teacher_quality_development_diagnostic",controls=controls,
        parent=str(PARENT.relative_to(ROOT)),parent_sha256=digest(PARENT),data_sha256=digest(DATA),
        states_sha256=digest(OLD/"states.json"),old_protocol_sha256=digest(OLD/"protocol.json"),
        parent_exposure_count=len(inherited),parent_validation_overlap=0,code_sha256=hashes,
        note="All teacher generation uses only frozen training states. Complete measured positive pools; no top-8 truncation. Equal gradient updates and starting weights/fresh Adam. Validation never supplies teachers.")
    if args.phase in ("prepare","all"):
        write(OUT/"protocol.json",protocol)
        (OUT/"pools").mkdir()
        for sample in training:
            record={**sample,"pools":{}}
            with _closed(state,sample,parent.network["colors"]):
                before=state.snapshot_domains()
                probes=learn.causal_candidates(state,parent.reward_colors,Random(29+sample["id"]),80,max_cores=64)
                for pool,cap,steps in (("direct",64,8192),("nested",256,65536)):
                    generator=ClusterCandidateGenerator(state.constraint_network,max_candidates=cap,max_steps=steps,
                        **{k:parent.config[k] for k in ("max_cores","max_open_colors","max_work","max_size")})
                    generate=generator.generate_nested if pool=="nested" else generator.generate
                    proposals=list(generate(state.domains,cyclic_only=True,compact_domains=True))
                    structural=len(proposals)
                    proposals+=probes
                    scored,seen=[],set()
                    def measure(candidate):
                        signature=learn._signature(candidate)
                        if signature in seen:
                            return
                        try:
                            parent.candidate_trace(candidate,state.domains)
                        except ValueError:
                            return
                        seen.add(signature)
                        metrics=learn.measure(parent,state,candidate,objective="pruning",known_satisfiable=sample["known_satisfiable"])
                        scored.append((candidate,metrics))
                    with torch.no_grad():
                        for candidate in proposals:
                            measure(candidate)
                        seeds=[c for c,m in scored if m["useful"]] or proposals
                        if seeds:
                            for candidate in learn.mutations(parent,state,seeds,Random(1029+sample["id"]),32):
                                measure(candidate)
                    teachers=learn._positive_teachers(scored,objective="pruning")
                    record["pools"][pool]=dict(structural=structural,probes=len(probes),measured=len(scored),teachers=[
                        dict(cores=c.core_keys,outputs=c.open_colors,metrics=m,
                             sequence_length=len(parent.candidate_trace(c,state.domains))) for c,m in teachers])
                    if before!=state.snapshot_domains():
                        raise AssertionError("Teacher probing changed frozen domains")
            write(OUT/"pools"/f"state_{sample['id']:02d}.json",record)
            print("prepared",sample["id"],sample["puzzle"],[(k,len(v["teachers"]),max((t["metrics"]["reward"] for t in v["teachers"]),default=0)) for k,v in record["pools"].items()],flush=True)
    elif json.loads((OUT/"protocol.json").read_text())!=json.loads(json.dumps(protocol)):
        raise AssertionError("Train phase requires identical frozen provenance/config")
    if args.phase=="prepare":
        return
    records=[json.loads((OUT/"pools"/f"state_{r['id']:02d}.json").read_text()) for r in training]
    shared=[r for r in records if all(r["pools"][p]["teachers"] for p in ("direct","nested"))]
    if not shared:
        raise AssertionError("No shared training states")
    order=[]
    rng=Random(29)
    while len(order)<240:
        visit=[r["id"] for r in shared]
        rng.shuffle(visit)
        order+=visit
    order=order[:240]
    stats={}
    for arm,(pool,fraction,_) in ARMS.items():
        sets=[r["pools"][pool]["teachers"] for r in shared]
        sets=[[t for t in ts if t["metrics"]["reward"]>=fraction*max(t["metrics"]["reward"] for t in ts)] for ts in sets]
        ts=[t for group in sets for t in group]
        stats[arm]=dict(pool_size=distribution([len(group) for group in sets]),
            rewards=distribution([t["metrics"]["reward"] for t in ts]),cores=distribution([len(t["cores"]) for t in ts]),
            sequence_length=distribution([t["sequence_length"] for t in ts]))
    write(OUT/"training_plan.json",dict(shared_state_ids=[r["id"] for r in shared],excluded_state_ids=[r["id"] for r in records if r not in shared],
        update_order=order,teacher_distributions=stats))
    print("teacher distributions",json.dumps(stats),flush=True)

    def evaluate(model,name,mode="trained"):
        roots=[]
        if mode=="trained":
            with torch.no_grad():
                for sample in validation:
                    with _closed(state,sample,model.network["colors"]):
                        candidate,_,_,trace=model.rollout(state.domains,greedy=True,beta=0.)
                        roots.append(dict(puzzle=sample["puzzle"],steps=len(trace),**learn.measure(model,state,candidate,
                            objective="pruning",known_satisfiable=True)))
        rows=[]
        for puzzle in ids:
            with torch.no_grad():
                row=solve(state.constraint_network,initial[puzzle],mode,model,nodes=10000,seconds=120,
                    cluster_attempts=32,baseline_candidates=32,baseline_steps=8192,stop_on_noop=False)
            if row["status"] in ("invalid","unsat"):
                raise AssertionError("Invalid validation solution")
            rows.append(dict(puzzle=puzzle,**row))
        complete=all(r["solved"] and r["valid"] for r in rows)
        result=dict(complete=complete,nodes=sum(r["nodes"] for r in rows) if complete else None,
            branch_free=sum(r["solved"] and r["nodes"]==1 for r in rows),greedy_useful=sum(r["useful"] for r in roots),
            greedy_states=len(roots),greedy_mean_reward=float(np.mean([r["reward"] for r in roots])) if roots else None,greedy_roots=roots,rows=rows)
        write(OUT/f"evaluation_{name}.json",result)
        print("evaluation",name,"nodes",result["nodes"],"branchfree",result["branch_free"],"greedy",result["greedy_useful"],"/",len(roots),flush=True)
        return result

    results={name:evaluate(parent,name,mode) for name,mode in (("plain","plain"),("heuristic32","heuristic"),("parent","trained"))}
    byid={r["id"]:r for r in shared}
    exposure=sorted(set(inherited)|instance_hashes(data,[initial[r["puzzle"]] for r in shared]))
    for arm in arm_order:
        if {str(p.relative_to(ROOT)):digest(p) for p in paths}!=hashes:
            raise AssertionError("Production code changed during controlled experiment")
        model=deepcopy(parent).train()
        if any(not torch.equal(v,parent.state_dict()[k]) for k,v in model.state_dict().items()):
            raise AssertionError("Starting weights changed")
        torch.manual_seed(29)
        optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
        rng,history,started=Random(29),[],perf_counter()
        pool,fraction,greedy=ARMS[arm]
        for update,idx in enumerate(order,1):
            record=byid[idx]
            teachers=teachers_for(record["pools"][pool])
            with _closed(state,record,model.network["colors"]):
                optimizer.zero_grad(set_to_none=True)
                loss=learn.teacher_loss(model,state.domains,teachers,rng,greedy=greedy,quality_fraction=fraction)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                history.append(dict(update=update,state=idx,loss=float(loss.detach()),gradient_norm=float(norm)))
            if update%24==0:
                print("train",arm,update,"/240 loss",history[-1]["loss"],flush=True)
        history_time=perf_counter()-started
        metadata=dict(method="controlled_teacher_quality_imitation",parent=str(PARENT),parent_sha256=digest(PARENT),
            train_instance_hashes=exposure,train_puzzles=sorted({r["puzzle"] for r in shared}),arm=arm,updates=240,
            seed=29,lr=.0003,fresh_optimizer=True,quality_fraction=fraction,greedy_imitation=greedy,
            protocol_sha256=digest(OUT/"protocol.json"),plan_sha256=digest(OUT/"training_plan.json"))
        checkpoint=OUT/f"{arm}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        _save(checkpoint,model,history,0.,1.,training=metadata,cost_metric="pruning_fraction")
        write(OUT/f"history_{arm}.json",dict(seconds=history_time,updates=240,rows=history))
        results[arm]=evaluate(model.eval(),arm)
    write(OUT/"summary.json",{name:{k:v for k,v in result.items() if k not in ("rows","greedy_roots")} for name,result in results.items()})


if __name__=="__main__":
    main()
