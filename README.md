Precision medicine aims to tailor treatments to the unique characteristics of individual patients. In this
paper, we develop a classification-based approach to estimate individualized treatment rule (ITR) by
leveraging both structured quantitative data and high-dimensional unstructured textual documents. To
tackle the challenge of incorporating text data, we propose an outcome-driven supervised nonnegative
matrix factorization method that extracts relevant topics for ITR estimation in a one-step procedure. Our
proposed method factorizes vectorized documents into a document-topic matrix and a topic-word matrix,
guided by the outcome. For estimation, we constructed a weighted and penalized objective function to
jointly estimate the document representation and ITR, solved by a projected gradient approach. Our
formulation enables the interpretability of the effect of the learned topics on ITR. We demonstrate the
performance of our method through simulation studies and real-world data from the MIMIC-IV intensive
care unit dataset.
