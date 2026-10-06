"""SWEEP-002 selection (pre-registered rule, speaker-disjoint validation only).
Rule: highest val certified fraction among configs within 1.0 pt of S0's val accuracy;
if none, smallest accuracy cost among configs certifying >= 55%."""
import glob, json
CFG = {  # tag -> env for retraining (FT/KD runs need that seed's control as INIT/TEACHER = {CTRL})
 "S1_cert03": "CERT_LAMBDA=0.3",
 "S2_ft_cert03": "INIT={CTRL} EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3",
 "S3_ft_cert03_kd": "INIT={CTRL} EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=0.3 TEACHER={CTRL}",
 "S4_ft_cert10_kd": "INIT={CTRL} EPOCHS=20 LR=5e-4 RAMP=1 CERT_LAMBDA=1.0 TEACHER={CTRL}",
 "S5_scratch_cert03_kd": "CERT_LAMBDA=0.3 TEACHER={CTRL}",
 "S6_ft_hub4_kd": "INIT={CTRL} EPOCHS=20 LR=5e-4 RAMP=1 NHUB=4 CERT_LAMBDA=0.3 TEACHER={CTRL}",
 "S7_ft_excap_kd": "INIT={CTRL} EPOCHS=20 LR=5e-4 RAMP=1 ALT=clamp TEACHER={CTRL}",
}
R = {}
for f in glob.glob("../results/sw2_*_s1.json"):
    r = json.load(open(f)); R[r["tag"]] = r
ctrl = 100 * R["S0_ctrl"]["acc_val"]
rows = sorted(((t, 100 * r["acc_val"], 100 * r["core_cert"], 100 * r["core_oracle"], r["violations"])
               for t, r in R.items()), key=lambda x: x[0])
print(f"{'tag':24s} {'val_acc':>8s} {'cost':>6s} {'cert':>6s} {'oracle':>6s} viol")
for t, a, c, o, v in rows:
    print(f"{t:24s} {a:8.2f} {a - ctrl:+6.2f} {c:6.1f} {o:6.1f} {v}")
cands = [(t, a, c) for t, a, c, o, v in rows if t != "S0_ctrl" and a >= ctrl - 1.0]
if cands:
    win = max(cands, key=lambda x: (x[2], x[1])); why = "within 1 pt, highest cert"
else:
    c55 = [(t, a, c) for t, a, c, o, v in rows if t != "S0_ctrl" and c >= 55]
    win = max(c55, key=lambda x: x[1]); why = "fallback: smallest cost with cert >= 55%"
print(f"\nWINNER: {win[0]} ({why}); val acc {win[1]:.2f} (control {ctrl:.2f}), cert {win[2]:.1f}%")
json.dump({"tag": win[0], "env": CFG[win[0]], "rule": why}, open("../results/sw2_winner.json", "w"), indent=1)
