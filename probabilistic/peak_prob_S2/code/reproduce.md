# 재현 명령 (프로젝트 루트에서, .venv_boosting)

```powershell
.venv_boosting\Scripts\python.exe -u -B -X utf8 notebooks\_shared\m1007_rare_peak_models.py --out reports\<새 run>   # 1차 M1·M2·M3
.venv_boosting\Scripts\python.exe -u -B -X utf8 notebooks\_shared\m1007_rare_peak_v2.py --out reports\<새 run>       # 2차 설정 탐색(v2)
.venv_boosting\Scripts\python.exe -u -B -X utf8 notebooks\_shared\m1007_rare_peak_v3.py --out reports\<새 run>       # 3차 S1·S2
```
v3는 v2 선정(reports/20261008T045108_KST_b05309_m1007_rare_peak_v2/v2_selection.json)과 잔차 확률 run(reports/20261007T235513_KST_36f6ca_m1007_residual_prob)을 읽는다.
입력 원천: data/processed/preprocessed_20261002/timeseries_15min_v1_previous_week.csv, 게이트 칸 OOF(reports/20261007T161906_KST_m1007_margin_gate, reports/20261007T163129_KST_m1007_margin_gate_long).
