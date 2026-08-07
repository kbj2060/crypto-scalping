# HuggingFace Offline Model Inventory - 2026-05-30

## Scope

This is an inventory only. No input feature contract, target, label, scoring
logic, or active `ai_*` replacement is decided here.

The next modeling step must explicitly choose:

- input feature families,
- prediction target / label,
- horizon,
- output prefix,
- where the output can be consumed.

Until then, these models are available candidates only.

## Runtime Packages

Checked in `/home/llewyn/miniconda3/envs/quant_ai`:

| Package | Available |
|---|---:|
| `transformers` | yes |
| `torch` | yes |
| `chronos` | yes |
| `gluonts` | yes |
| `uni2ts` | yes |
| `timesfm` | no |

Implication: TimesFM model files are cached, but the Python package is not
installed in `quant_ai` yet. Chronos and Transformers-based models are directly
loadable offline.

## Cached Candidate Models

Local HuggingFace cache root:

```text
/home/llewyn/.cache/huggingface/hub
```

| Candidate | Local cache status | Notes before use |
|---|---|---|
| `ibm/patchtsmixer-etth1-pretrain` | cached | Good candidate for sequence representation or fine-tuned direction/regression head. Objective not decided. |
| `amazon/chronos-t5-tiny` | cached | Directly loadable through `chronos`. Forecast-distribution candidate. |
| `amazon/chronos-bolt-tiny` | cached | Chronos-family candidate; loading/API check still needed before use. |
| `amazon/chronos-2` | cached | Chronos-family candidate; loading/API check still needed before use. |
| `Salesforce/moirai-1.0-R-small` | cached | `uni2ts` package is available; exact inference wrapper still needs a smoke check. |
| `Salesforce/moirai-2.0-R-small` | cached | `uni2ts` package is available; exact inference wrapper still needs a smoke check. |
| `google/timesfm-1.0-200m` | cached | Package missing. Do not use until `timesfm` runtime is installed or a local wrapper is proven. |
| `google/timesfm-1.0-200m-pytorch` | cached | Package missing. Do not use until runtime is installed/proven. |
| `google/timesfm-2.0-500m-pytorch` | cached | Package missing. Do not use until runtime is installed/proven. |
| `google/timesfm-2.5-200m-pytorch` | cached | Package missing. Do not use until runtime is installed/proven. |
| `ibm-granite/granite-timeseries-ttm-r1` | cached | Transformers time-series candidate; loading/API check needed. |
| `ibm-granite/granite-timeseries-ttm-v1` | cached | Transformers time-series candidate; loading/API check needed. |
| `mldi-lab/Kairos_23m` | cached | Candidate model; loading/API check needed. |
| `NeoQuasar/Kronos-small` | cached | Candidate model; tokenizer caches are also present. Loading/API check needed. |
| `NeoQuasar/Kronos-Tokenizer-2k` | cached | Tokenizer/cache component. |
| `NeoQuasar/Kronos-Tokenizer-base` | cached | Tokenizer/cache component. |
| `time-series-foundation-models/Lag-Llama` | cached | GluonTS/Lag-Llama candidate; loading/API check needed. |

## Guardrails

- Do not generate new `ai_*`, `ai_hf_*`, or other feature columns until the
  input/target contract is chosen.
- Do not overwrite existing `data/nf_*` NeuralForecast artifacts.
- Do not use `teacher_*`, `m7_*`, `a5dir_*`, or downstream regime/router outputs
  as inputs for a new upstream AI feature generator unless a new versioned
  artifact and no-leak audit are explicitly approved.
- Offline loading must use local cache only. Missing cache or missing runtime
  package is a hard blocker, not a silent fallback.

