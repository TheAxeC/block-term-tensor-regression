# Block-Term Tensor Regression (BTTR)

BTTR is a deflation-based method in which the maximally correlated representations of ***X*** and ***Y*** are extracted via ACE/ACCoS (Automatic Component Extraction / Automatic Correlated Component Selection) at each iteration. Therefore, BTTR inherits the advantages of the proposed ACE/ACCoS and does not require one to set the model parameters manually. This provides BTTR with an additional important property: the ability to model complex data in which the optimal Multilinear Rank (MTR) is not necessarily stable across sequential decompositions.

[1] Faes, Axel, Flavio Camarrone, and Marc M. Van Hulle. "Single finger trajectory prediction from intracranial brain activity using block-term tensor regression with fast and automatic component extraction." IEEE Transactions on Neural Networks and Learning Systems (2022).
