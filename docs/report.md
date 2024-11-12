# Report

Once you have computed the results for the models under evaluation, you can inspect and compare their metric scores by relying on the `Report` class.
Moreover, you can export reports as [`LaTeX`](https://en.wikipedia.org/wiki/LaTeX) tables for scientific publications.

```python
from guardbench import Report
from guardbench.datasets import get_datasets_by

report = Report(
    models=[  # Models under comparison
        {"name": "Llama Guard", "alias": "LG"},
        {"name": "Llama Guard 2", "alias": "LG-2"},
        {"name": "Llama Guard Defensive", "alias": "LG-D"},
        {"name": "Llama Guard Permissive", "alias": "LG-P"},
        {"name": "MD-Judge", "alias": "MD-J"},
        {"name": "Mistral", "alias": "Mis"},
        {"name": "Mistral Plus", "alias": "Mis+"},
    ],
    datasets=[  # Chosen evaluation datasets
        "malicious_instruct",
        "do_not_answer",
        "xstest",
        "openai_moderation_dataset",
        "beaver_tails_330k",
        "harmful_qa",
        "prosocial_dialog",
    ],
    out_dir="results",  # Where results are stored
)
```

You can display the report in `IPython` notebooks as follows:
```python
report.display()
```

Output:

| Dataset                   | Metric | LG    | LG-2      | LG-D             | LG-P             | MD-J             | Mis   | Mis+             |
| ------------------------- | ------ | ----- | --------- | ---------------- | ---------------- | ---------------- | ----- | ---------------- |
| MaliciousInstruct         | Recall | 0.820 | 0.890     | **1.000**        | 0.920            | <ins>0.990</ins> | 0.980 | <ins>0.990</ins> |
| DoNotAnswer               | Recall | 0.321 | 0.442     | <ins>0.496</ins> | 0.399            | **0.501**        | 0.435 | 0.460            |
| XSTest                    | F1     | 0.819 | **0.891** | 0.783            | 0.812            | 0.858            | 0.829 | <ins>0.878</ins> |
| OpenAI Moderation Dataset | F1     | 0.744 | 0.761     | 0.658            | 0.756            | <ins>0.774</ins> | 0.722 | **0.779**        |
| BeaverTails 330k          | F1     | 0.686 | 0.755     | <ins>0.778</ins> | 0.755            | **0.887**        | 0.696 | 0.740            |
| HarmfulQA                 | F1     | 0.171 | 0.391     | **0.764**        | 0.563            | <ins>0.676</ins> | 0.648 | 0.427            |
| ProsocialDialog           | F1     | 0.519 | 0.383     | **0.792**        | 0.691            | 0.720            | 0.697 | <ins>0.762</ins> |
| Wins                      |        | 0     | 1         | 3                | 0                | 3                | 0     | 1                |


You can export the report to `LaTeX` as follows:
```python
report.to_latex()
```

Output:

```latex
\begin{table*}[!ht]
\centering
\begin{tabular}{lllllllll}
\hline
 Dataset                   & Metric   & LG                & LG-2              & LG-D              & LG-P              & MD-J              & Mis               & Mis+              \\
\hline
 MaliciousInstruct         & Recall   &            0.820  &            0.890  &    \textbf{1.000} &            0.920  & \underline{0.990} &            0.980  & \underline{0.990} \\
 DoNotAnswer               & Recall   &            0.321  &            0.442  & \underline{0.496} &            0.399  &    \textbf{0.501} &            0.435  &            0.460  \\
 XSTest                    & F1       &            0.819  &    \textbf{0.891} &            0.783  &            0.812  &            0.858  &            0.829  & \underline{0.878} \\
 OpenAI Moderation Dataset & F1       &            0.744  &            0.761  &            0.658  &            0.756  & \underline{0.774} &            0.722  &    \textbf{0.779} \\
 BeaverTails 330k          & F1       &            0.686  &            0.755  & \underline{0.778} &            0.755  &    \textbf{0.887} &            0.696  &            0.740  \\
 HarmfulQA                 & F1       &            0.171  &            0.391  &    \textbf{0.764} &            0.563  & \underline{0.676} &            0.648  &            0.427  \\
 ProsocialDialog           & F1       &            0.519  &            0.383  &    \textbf{0.792} &            0.691  &            0.720  &            0.697  & \underline{0.762} \\
\hline
 Wins                      &          &            0      & \underline{1}     &    \textbf{3}     &            0      &    \textbf{3}     &            0      & \underline{1}     \\
\hline
\end{tabular}
\caption{Evaluation results. Best results are highlighted in boldface. Second-best results are underlined.}
\label{tab:results}
\end{table*}

```