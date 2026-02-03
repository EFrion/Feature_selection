The authors of [1] developed a way for investors to time their entry and exit in stock markets of emerging countries. The key point is that technical indicators timing the market in emerging and developed countries are different. For instance, they note that "the volume category for the emerging markets ETFs is Archer's on balance volume (AOBV), while on the developed market ETF is price volume rank (PVR)[...]; this supports the general idea that emerging markets' predictions use quantitative features, while developed markets rely more on qualitative features" (Sagaceta-Mejía et al., 10). Thus the use of technical indicators seems justified. They claim that their method helps investors to choose the most relevant indicators for these markets, effectively reducing dimensionality, and therefore reducing computational
time while improving accuracy at the same time. This code aims at reproducing the main results of their paper. Focusing on the Chilean ETF names ECH, and selecting features through Pearson's correlation coefficient, we find results in agreement with the author.

| n | 0 (all features) | 1 | 2 | 3 | 4 |
| :---: | :---: | :---: | | :---: | :---: | :---: |
| Features | 324 | 78 | 20 | 5 | 2 |
| :---: | :---: | :---: | | :---: | :---: | :---: |
| Accuracy (%) | 68.31 | 72.34 | 79.95 | 80.45 | 80.45 |

![ECH accuracy gain](data/ECH_accuracy_gain_example.png)

[1] Sagaceta-Mejía, Alma Rocío, et al. “An Intelligent Approach for Predicting Stock Market Movements in Emerging Markets Using Optimized Technical Indicators and Neural Networks.” Economics, vol. 18, no. 1, 2024, pp. 20220073.
