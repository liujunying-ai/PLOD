# PLOD
This repository provides the MATLAB implementation for the following article:

J.-Y. Liu, J.-P. Sun, Y.-H. Zhao, B.-B. Jia, M.-L. Zhang. [Partial Label Learning with Semi-Supervised Clustering Disambiguation](https://doi.org/10.1016/j.patcog.2025.112138). In: Pattern Recognition, vol.171: Article 112138, 2026.

"PRA_PL_LEAF.m", "PRA_PL_AGGD.m" and "PRA_PL_CL.m" are the main functions of the three base classifiers, which implement the PRA-enhanced versions of PL-LEAF, PL-AGGD and PL-CL, respectively. "ReadMe_PLLEAF.m", "ReadMe_PLAGGD.m" and "ReadMe_PLCL.m" are exemplar files on how the PRA_PL_LEAF, PRA_PL_AGGD and PRA_PL_CL programs could be used, respectively.

### 📁 Base Classifier Source Code

&zwnj;**NOTE:**&zwnj; In addition to the PRA implementation, this repository contains three folders — `PL_LEAF/`, `PL_AGGD/`, and `PL_CL/` — each holding the source code of one base classifier. The original implementations were obtained from the following links:

| Method   | Original Source |
|----------|-----------------|
| PL-LEAF  | <https://palm.seu.edu.cn/zhangml/files/PL-LEAF.rar>{target="_blank"} |
| PL-AGGD  | <https://palm.seu.edu.cn/zhangml/files/PL-AGGD.rar>{target="_blank"} |
| PL-CL    | <https://palm.seu.edu.cn/zhangml/files/PLCL.rar>{target="_blank"}    |

We have made necessary modifications to the source code in order to integrate them with the PRA framework.
