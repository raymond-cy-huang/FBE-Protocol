# Background Fidelity Protocol Validation

This experiment is a Section 4.2 smoke run for FBE-Protocol v2.

The pSp frontalization outputs are used as a benchmark scenario with known
background fidelity degradation. The goal is to validate that the FBE
evaluation module detects and quantifies background degradation; this is not a
pSp model evaluation.

## Sample Run

The script samples 10 paired cases total from the input folder and writes
`bg_fideility_result.csv` and `bg_fideility_summarization.csv` to the configured
output folder. Gender folders are used only to find paired inputs; the summary
row is `overall`.

```bash
conda run -n fbe-protocol python experiments/background_fideility_protocol_validation/run_sample.py
```

For the full Section 4.2 run after the sample is approved:

```bash
conda run -n fbe-protocol python experiments/background_fideility_protocol_validation/run_sample.py --sample-size 0
```

Default input:

```text
D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start\exper_05_psp_face_frontal_analysis\eval_bg_iou
```

Default output:

```text
D:\Ph.D\01_Experiments_GAN_Inv_Log\2026.03.08_All_Experiments_Start\FBE-Protocol_final_output\4.2_Background_Fidelity_Protocol_Validation（ＷＳＬ）
```
