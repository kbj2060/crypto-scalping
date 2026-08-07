"""BTC-110 CUSUM event model with calibrated long/short net-expectancy decisions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_event_labels import causal_cusum_events  # noqa: E402
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from train_eval_btc110_cusum_tb_causal_20260804 import atr_move  # noqa: E402
from train_eval_btc_110branch_causal_20260804 import COST, HORIZON, LEVERAGE, MARGIN, load_frame  # noqa: E402

OUT = ROOT / "tmp/btc110_expectancy_causal_20260804"
TRAIN_END, VAL_END, CAL_END, TEST_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
CUSUM_MULTS, EXPECTANCY_FLOORS = [1.5, 2.0, 2.5], [0.0, .0005, .0010, .0015]
TP_MULT, SL_MULT, MIN_TP, MIN_SL, MIN_TRADES = 1.2, .8, .006, .004, 30
BOOTSTRAP_SAMPLES, DEVICE = 2000, "cuda" if torch.cuda.is_available() else "cpu"


class ExpectancyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def branch(n, width): return nn.Sequential(nn.Linear(n, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(.1))
        self.market, self.context = branch(94, 64), branch(16, 32)
        self.fuse = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Dropout(.1))
        self.residual, self.norm = nn.Linear(64, 64), nn.LayerNorm(64)
        self.long_head, self.short_head = nn.Linear(64, 1), nn.Linear(64, 1)

    def forward(self, x):
        z = self.fuse(torch.cat([self.market(x[:, :94]), self.context(x[:, 94:])], 1))
        z = torch.nn.functional.gelu(self.norm(z + self.residual(z)))
        return self.long_head(z).squeeze(1), self.short_head(z).squeeze(1)


def side_labels(frame: pd.DataFrame, events: np.ndarray, atr: np.ndarray):
    high, low, op = (frame[c].to_numpy(float) for c in ("high", "low", "open")); kept=[]; labels=[]
    for i in events:
        entry_i, end_i = int(i) + 1, int(i) + HORIZON
        if end_i >= len(frame): continue
        entry, tp, sl = op[entry_i], max(MIN_TP, TP_MULT * atr[i]), max(MIN_SL, SL_MULT * atr[i])
        long_y = short_y = 0
        for hi, lo in zip(high[entry_i:end_i + 1], low[entry_i:end_i + 1]):
            long_tp, long_sl = hi >= entry * (1 + tp), lo <= entry * (1 - sl)
            short_tp, short_sl = lo <= entry * (1 - tp), hi >= entry * (1 + sl)
            if not (long_tp and long_sl) and long_tp: long_y = 1
            if not (short_tp and short_sl) and short_tp: short_y = 1
            if long_tp or long_sl or short_tp or short_sl: break
        kept.append(i); labels.append([long_y, short_y])
    return np.asarray(kept, int), np.asarray(labels, np.float32)


def run_epoch(model, loader, optimiser=None):
    model.train(optimiser is not None); total=n=0
    for x, y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        if optimiser: optimiser.zero_grad()
        long, short = model(x); loss=(nn.functional.binary_cross_entropy_with_logits(long,y[:,0])+nn.functional.binary_cross_entropy_with_logits(short,y[:,1]))/2
        if optimiser: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimiser.step()
        total += loss.item(); n += 1
    return total/max(n,1)


def logits(model, x):
    model.eval(); out=[]
    with torch.no_grad():
        for (b,) in DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=1024):
            a,z=model(b.to(DEVICE)); out.append(torch.stack([a,z],1).cpu().numpy())
    return np.concatenate(out)


def calibrated_probs(raw_logits, calibrators):
    raw = 1/(1+np.exp(-raw_logits)); return np.column_stack([calibrators[i].predict(raw[:,i]) for i in range(2)])


def bootstrap_lower(ledger: pd.DataFrame) -> float:
    returns=ledger.trade_return.to_numpy(float)
    if len(returns)==0: return float("-inf")
    rng=np.random.default_rng(20260804); means=rng.choice(returns,(BOOTSTRAP_SAMPLES,len(returns)),replace=True).mean(1)
    return float(np.quantile(means,.025)*100)


def evaluate(frame, events, probabilities, atr, floor):
    tp=np.maximum(MIN_TP,TP_MULT*atr[events]); sl=np.maximum(MIN_SL,SL_MULT*atr[events])
    long_e=probabilities[:,0]*tp-(1-probabilities[:,0])*sl-COST
    short_e=probabilities[:,1]*tp-(1-probabilities[:,1])*sl-COST
    score=np.where(long_e>=short_e,long_e,-short_e)
    result=simulate_single_position(timestamps=frame.timestamp,open_px=frame.open.to_numpy(),high=frame.high.to_numpy(),low=frame.low.to_numpy(),close=frame.close.to_numpy(),decision_indices=events,scores=score,tp_moves=tp,sl_moves=sl,upper_threshold=floor,lower_threshold=-floor,horizon_bars=HORIZON,margin_fraction=MARGIN,leverage=LEVERAGE,roundtrip_cost_rate=COST)
    m=bar_level_performance(result.equity,result.ledger);m["mean_trade_return_pct"]=float(result.ledger.trade_return.mean()*100) if len(result.ledger) else 0.;m["skipped_while_open"]=result.skipped_while_open;m["bootstrap_mean_return_lower95_pct"]=bootstrap_lower(result.ledger)
    return m,result.ledger


def main():
    OUT.mkdir(parents=True,exist_ok=True);frame,cols=load_frame();ts=pd.DatetimeIndex(frame.timestamp);atr=atr_move(frame);raw=frame[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
    masks={"train":purged_decision_mask(ts,start=ts[0],end=TRAIN_END,horizon_bars=HORIZON),"val":purged_decision_mask(ts,start=TRAIN_END,end=VAL_END,horizon_bars=HORIZON),"cal":purged_decision_mask(ts,start=VAL_END,end=CAL_END,horizon_bars=HORIZON),"test":purged_decision_mask(ts,start=CAL_END,end=TEST_END,horizon_bars=HORIZON)}
    train_rows=np.flatnonzero(masks["train"]&np.isfinite(raw).all(1));mean,std=raw[train_rows].mean(0),raw[train_rows].std(0);std[std<1e-6]=1;x=np.clip((raw-mean)/std,-10,10).astype(np.float32)
    rows=[];eligible=[];models={}
    for mult in CUSUM_MULTS:
        events,labels=side_labels(frame,causal_cusum_events(frame.close.to_numpy(),atr,mult),atr);good=np.isfinite(raw[events]).all(1);events,labels=events[good],labels[good];groups={k:np.flatnonzero(masks[k][events]) for k in masks}
        model=ExpectancyNet().to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);train=DataLoader(TensorDataset(torch.from_numpy(x[events[groups['train']]]),torch.from_numpy(labels[groups['train']])),batch_size=256,shuffle=True);val=DataLoader(TensorDataset(torch.from_numpy(x[events[groups['val']]]),torch.from_numpy(labels[groups['val']])),batch_size=512)
        best,bad,state,val_loss=float('inf'),0,None,None
        for epoch in range(1,13):
            tr,va=run_epoch(model,train,opt),run_epoch(model,val);print(f"cusum={mult} epoch={epoch} train_bce={tr:.5f} val_bce={va:.5f}",flush=True)
            if va<best-1e-5:best,bad,state,val_loss=va,0,{k:v.cpu().clone() for k,v in model.state_dict().items()},va
            else:
                bad+=1
                if bad>=3:break
        model.load_state_dict(state);val_logits=logits(model,x[events[groups['val']]]); raw_p=1/(1+np.exp(-val_logits)); calibrators=[IsotonicRegression(out_of_bounds='clip').fit(raw_p[:,i],labels[groups['val'],i]) for i in range(2)];cal_p=calibrated_probs(logits(model,x[events[groups['cal']]]),calibrators);models[mult]=(state,calibrators,events,groups)
        for floor in EXPECTANCY_FLOORS:
            m,ledger=evaluate(frame,events[groups['cal']],cal_p,atr,floor);row={"cusum_multiplier":mult,"expectancy_floor":floor,"validation_bce":val_loss,"event_counts":{k:int(len(v)) for k,v in groups.items()},**m};rows.append(row)
            if m['pnl']>0 and m['trades']>=MIN_TRADES and m['bootstrap_mean_return_lower95_pct']>0:eligible.append((m['pnl'],row,ledger))
    pd.DataFrame(rows).to_json(OUT/'calibration_candidates.json',orient='records',indent=2);report={"architecture":"btc110_calibrated_dual_expectancy_event_model","layers":{"market":"94→64→LayerNorm→GELU→Dropout(0.1)","context":"16→32→LayerNorm→GELU→Dropout(0.1)","fusion":"96→64→GELU→Dropout(0.1)→residual(64)→LayerNorm→GELU","output":"64→long TP-first logit; 64→short TP-first logit"},"selection_rule":"CAL only: positive PnL, >=30 trades, bootstrap lower-95 mean trade return > 0","calibration_candidates":rows,"contracts":{"fresh_forward_bar_by_bar":True,"thresholds_fit_on_calibration_only":True,"trade_ledgers_used_as_input":False,"saved_parent_exit_timestamps_used":False,"future_rows_used_for_entry":False,"split_targets_purged":True,"single_position":True,"bar_level_mark_to_market":True,"probability_calibrator_fit_on_validation_only":True,"test_used_for_selection":False},"promotion_eligible":False}
    if not eligible:
        report['result']='NO_CALIBRATION_CANDIDATE_PASSED_GATE';report['test_metrics']=None;report['promotion_blockers']=['no positive, statistically supported calibration candidate'];(OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(report['result']);return 0
    _,chosen,cal_ledger=max(eligible,key=lambda x:x[0]);mult,floor=chosen['cusum_multiplier'],chosen['expectancy_floor'];state,calibrators,events,groups=models[mult];model=ExpectancyNet().to(DEVICE);model.load_state_dict(state);test_p=calibrated_probs(logits(model,x[events[groups['test']]]),calibrators);test_metric,test_ledger=evaluate(frame,events[groups['test']],test_p,atr,floor);cal_ledger.to_csv(OUT/'selected_calibration_ledger.csv',index=False);test_ledger.to_csv(OUT/'test_ledger.csv',index=False);report.update({"result":"CALIBRATION_GATE_PASSED","selected_config":chosen,"test_metrics":test_metric,"promotion_blockers":["test period previously inspected","research-only artifact lineage"]});(OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps({"selected":chosen,"test":test_metric},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
