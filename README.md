# AraC2AIdet


\begin{algorithm}[H]
\caption{Cross-Domain Stacked Ensemble Framework for AI-Generated Text Detection}
\label{alg:stacking}

\begin{algorithmic}[1]
\Require Social media dataset $D_{soc}$, Academic abstracts dataset $D_{abs}$, 
         Base models $\mathcal{B} = \{$AraBERT, AraELECTRA, DeBERTa, XLM-RoBERTa$\}$,
         Meta-learners $\mathcal{M} = \{$LR, SVM, RF, XGBoost$\}$,
         Number of folds $K=5$
\Ensure Final cross-domain predictions $\hat{y}$

\Function{BuildDatasets}{$D_{human}, D_{llm}$}
    \State Combine $D_{human}$ with $D_{llm}$ to construct a balanced binary dataset
    \State \Return $D_{comb}$
\EndFunction

\Function{TrainBaseModels}{$D_{train}, \mathcal{B}, K$}
    \For{each base model $b \in \mathcal{B}$}
        \State Apply $K$-fold cross-validation on $D_{train}$
        \State Collect out-of-fold predictions $\rightarrow X_b$
    \EndFor
    \State Concatenate $\{X_b\}_{b \in \mathcal{B}} \rightarrow X_{train}$
    \State \Return $X_{train}$
\EndFunction

\Function{TestRepresentation}{$D_{test}, \mathcal{B}$}
    \For{each base model $b \in \mathcal{B}$}
        \State Train $b$ on the full $D_{train}$
        \State Predict probabilities on $D_{test}$ $\rightarrow X_b$
    \EndFor
    \State Concatenate $\{X_b\}_{b \in \mathcal{B}} \rightarrow X_{test}$
    \State \Return $X_{test}$
\EndFunction

\Statex
\State \textbf{Direction 1: Social $\rightarrow$ Abstracts}
\For{each LLM $g \in \{$ALLaM, LLaMA, Jais, OpenAI$\}$}
    \State $D_{train} \gets$ \Call{BuildDatasets}{Social(Human), Social($g$)}
    \State $X_{train} \gets$ \Call{TrainBaseModels}{$D_{train}, \mathcal{B}, K$}
    \For{each test subset $T \in \{$Human, ALLaM, LLaMA, Jais, OpenAI$\}$ of $D_{abs}$}
        \State $X_{test} \gets$ \Call{TestRepresentation}{$T, \mathcal{B}$}
        \For{each meta-learner $m \in \mathcal{M}$}
            \State Train $m$ on $X_{train}$
            \State Predict $\hat{y} = m(X_{test})$
            \State Evaluate performance
        \EndFor
    \EndFor
\EndFor

\Statex
\State \textbf{Direction 2: Abstracts $\rightarrow$ Social}
\For{each LLM $g \in \{$ALLaM, LLaMA, Jais, OpenAI$\}$}
    \State $D_{train} \gets$ \Call{BuildDatasets}{Abstract(Human), Abstract($g$)}
    \State $X_{train} \gets$ \Call{TrainBaseModels}{$D_{train}, \mathcal{B}, K$}
    \For{each test subset $T \in \{$Human, ALLaM, LLaMA, Jais, OpenAI$\}$ of $D_{soc}$}
        \State $X_{test} \gets$ \Call{TestRepresentation}{$T, \mathcal{B}$}
        \For{each meta-learner $m \in \mathcal{M}$}
            \State Train $m$ on $X_{train}$
            \State Predict $\hat{y} = m(X_{test})$
            \State Evaluate performance
        \EndFor
    \EndFor
\EndFor

\end{algorithmic}
\end{algorithm}

