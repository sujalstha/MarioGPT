from pathlib import Path
import json, numpy as np
root=Path(__file__).parent
rows=json.loads((root/'results/summary.json').read_text())
manifest=json.loads((root/'results/manifest.json').read_text())
assert len(rows)==50 and all(r['status']=='complete' for r in rows)
table=[]
for arch in ['transformer','diffusion']:
 for k in [3,6,9,12,15]:
  rs=[r for r in rows if r['architecture']==arch and r['k']==k];assert len(rs)==5
  def cell(key,scale=100):
   x=np.array([r[key] for r in rs])*scale
   return f'{x.mean():.1f} $\\pm$ {x.std(ddof=1):.1f}'
  table.append(f'{arch.title()} & {k} & {cell("copy_rate")} & {cell("heldout_copy_rate")} & {cell("reachability")} \\\\')
k3={a:np.mean([r['copy_rate'] for r in rows if r['architecture']==a and r['k']==3])*100 for a in ['transformer','diffusion']}
body=r'''\documentclass[final]{alt2027}
\usepackage{times,booktabs}
\title[Mario Level Memorization]{Midterm Progress Report (Track D, implementation-based):\\Do Generative Level Models Copy Their Training Data?}
\altauthor{\Name{Sujal Shrestha} \Email{sujalshrestha@ou.edu}\\
\Name{Jackson McKay} \Email{jackson.j.mckay-1@ou.edu}\\
\addr CS 4713, Fall 2026\\
Github: \url{https://github.com/sujalstha/MarioGPT}}
\begin{document}
\maketitle
\begin{abstract}
A playable generated Mario level may resemble a training level. We ask how often small Transformer and diffusion generators produce near-copies, and whether copying decreases as the training set grows. We trained 50 models on matched subsets of 3, 6, 9, 12, or 15 levels, with five seeds per setting, and generated 1,000 slices per model. At three levels, mean near-copy rates were TRANSRATE\% for the Transformer and DIFFRATE\% for diffusion. These are measurements from short, fixed training budgets rather than converged reproductions. Low copying alone cannot establish generalization: our held-out baseline and local reachability check help distinguish novelty from poor generation.
\end{abstract}
\begin{keywords}Track D, theory of modern AI; implementation-based study; memorization, level generation\end{keywords}
\section{Introduction}
Our original proposal compares MarioGPT \citep{mariogpt} with unconditional diffusion level generation \citep{diffusion}. MarioGPT fine-tunes a DistilGPT2-based model on tile sequences, uses cross-attention for prompts, and evaluates generated levels with an A* agent. Importantly, its Section 4.3 already examines memorization through nearest-training examples and sampling temperature. Lee and Simo-Serra use a U-Net diffusion model with attention and one-hot tile representations. Our contribution is a matched, quantitative training-set-size sweep using a fixed near-copy threshold, not the first observation that level generators can copy.

We test the proposal's hypothesis that more than half the generated slices are near-copies at three training levels and that the rate falls as data increases. The course connection is overfitting: fitting a tiny corpus need not imply useful generalization. Our experiments measure a behavior; they do not prove a generalization bound.

\section{Implementation and Results}
\textbf{Setup.} We use NUMLEVELS VGLC maps from Super Mario Bros. and The Lost Levels, with a fixed five-level holdout selected before slicing. Maps are ground-aligned to 14 rows by cropping or padding above. We extract $14\times16$ slices with stride 1. For each of five seeds, one random level ordering gives nested subsets $k\in\{3,6,9,12,15\}$, shared by both architectures. The resulting $2\times5\times5=50$ trained models produce 50,000 slices. Exact level IDs, losses, outputs, and settings are saved in the private repository; \texttt{mlciv} has been invited.

Our Transformer retains pretrained DistilGPT2 blocks but replaces the language vocabulary with tile IDs and a start token. It omits text prompts and novelty search. We train 300 updates with batch 16 and AdamW (learning rate $5\times10^{-5}$, no weight decay), then sample at temperature 1 without top-$k$ truncation. Our small U-Net uses 32/64/32 channels, one downsampling stage, a skip connection, and a scalar time channel. It trains for 1,500 updates with batch 16 and Adam ($10^{-3}$), predicting noise over 100 linear-beta DDPM steps; tiles are decoded by argmax. It omits the diffusion paper's attention, and uses our proposal's $14\times16$ format rather than that paper's $14\times14$. Training budgets are equal across dataset sizes within each architecture, not across architectures.

\textbf{Experiment Set 1: copying.} For each output $x$, we find its exact nearest training slice under tilewise Hamming distance, $d_{\min}(x)=\min_{s\in S}\sum_{i=1}^{224}\mathbf1[x_i\ne s_i]$. The copy rate is the fraction with $d_{\min}\le11$ (at most 5\% of tiles). We also save exact-copy rates and distances. Table~\ref{tab:results} and Figure~\ref{fig:metrics} report the completed grid. A larger corpus supplies more possible matches, so we evaluate real held-out slices against each training subset as a reference. The five training runs are the repeats; the 1,000 outputs within a run are not independent training experiments.

\begin{table}[t]
\centering\small
\caption{Mean $\pm$ sample standard deviation over five seeds. Rates are percentages. Reachability is a local proxy, not an A* score.}\label{tab:results}
\begin{tabular}{lrrrr}\toprule
Model & Levels & Near-copy & Held-out near-copy & Reachable\\\midrule
TABLEROWS
\bottomrule\end{tabular}
\end{table}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{../figures/metrics.pdf}
\caption{Copy rate, nearest-training distance, and local reachability versus training-set size. Error bars show sample standard deviation over five seeds. The dashed curve is the held-out real-level baseline.}\label{fig:metrics}
\end{figure}

\textbf{Experiment Set 2: reachability after filtering.} We search supported landing positions connected by discretized jump arcs up to four tiles horizontally, with heights 1--4. Enemies block the path; empty space and coins are passable. This point-agent check tests left-to-right traversal of a slice with approximate collisions. We recompute it on outputs with $d_{\min}>11$ and save conditional rates and counts. It does not reproduce Mario's physics or establish full-level playability. Figure~\ref{fig:examples} shows the closest and median-distance outputs at $k=3$, seed 0, beside their nearest training slices; this selection is deterministic.

\begin{figure}[t]\centering
\includegraphics[width=.88\linewidth]{../figures/examples.pdf}
\caption{Tile-coded generated slices and nearest training matches. Each row shows the minimum-distance and median-distance examples for one model. Colors consistently identify tile categories; these are data visualizations, not game screenshots.}\label{fig:examples}
\end{figure}

The three-level rates are compared with our predeclared 50\% hypothesis, but undertraining is an alternative explanation for low copying. We therefore do not interpret novelty as proof that either architecture learned playable structures. Differences also reflect model capacity and optimization budgets, so this is not a ranking of the published methods.

\section{Difficulties So Far}
\begin{itemize}\setlength{\itemsep}{0pt}
\item \textbf{Small corpus and inconsistent heights.} Base-game data alone contains only 15 maps. Adding Lost Levels supports the full grid plus held-out evaluation, but requires explicit height normalization. A validation failure caught this before main training.
\item \textbf{Short training and generation setup.} We imposed a 45-GPU-minute cap, used fixed update budgets, and corrected an attention-mask warning in the pilot. Pilot results are excluded. Neither model is claimed to have converged.
\item \textbf{Metric interpretation.} Common empty-space and ground patterns can cause matches in unseen levels. Our reachability proxy is also coarse. Boundary tests at 0, 11, and 12 changed tiles and simple traversable/blocked cases passed; saved distances are recomputed for verification.
\end{itemize}

\section{Plan for the Rest of the Semester}
\begin{itemize}\setlength{\itemsep}{0pt}
\item \textbf{Nov. 10--25:} Increase training budgets after approval, inspect loss convergence, test sampling temperatures, and improve the playability evaluation while keeping the same copy metric and matched data splits.
\item \textbf{Nov. 26--Dec. 7:} Analyze copying versus training exposure and model size, inspect examples, and connect findings to overfitting and Occam's principle.
\item \textbf{Dec. 8/10 and Dec. 14:} Present the project, then submit the final report and code. Before then, meet Dr. Cao for the required 15-minute progress discussion; this meeting is not yet completed.
\end{itemize}

\section{What Each of Us Has Done}
\textbf{Sujal Shrestha:} Supplied the original team proposal, selected the Mario topic and report reference, and set the compute limit. \textbf{Jackson McKay:} No completed individual technical contribution was reported at drafting time. \textbf{Assistance:} Codex implemented and ran the experiments, produced figures, and drafted this report at Sujal's direction. Both students still need to review and understand the code, results, and paper connections before submission; implementation work is not attributed to them without verification.

\begin{thebibliography}{2}
\bibitem[Sudhakaran et~al.(2023)]{mariogpt} S. Sudhakaran, M. Gonz\'alez-Duque, C. Glanois, M. Freiberger, E. Najarro, and S. Risi. MarioGPT: Open-Ended Text2Level Generation through Large Language Models. \emph{NeurIPS}, 2023.
\bibitem[Lee and Simo-Serra(2023)]{diffusion} H. J. Lee and E. Simo-Serra. Using Unconditional Diffusion Models in Level Generation for Super Mario Bros. \emph{MVA}, 2023.
\end{thebibliography}
\end{document}
'''
body=body.replace('TRANSRATE',f'{k3["transformer"]:.1f}').replace('DIFFRATE',f'{k3["diffusion"]:.1f}').replace('NUMLEVELS',str(len(manifest['files']))).replace('TABLEROWS','\n'.join(table))
(root/'alt2027-template/CLT_Midterm_Report.tex').write_text(body)
