"""
Functions for principal component analysis (PCA)

---------------------------------------------------------------------

pca
    principal component analysis (singular value decomposition)

---------------------------------------------------------------------

Copyright 2018 Facebook Inc.
Copyright 2019 Evgenii Zheltonozhskii
All rights reserved.

"Software" means the fbpca software distributed by Facebook Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

* Neither the name Facebook nor the names of its contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Additional grant of patent rights:

Facebook hereby grants you a perpetual, worldwide, royalty-free,
non-exclusive, irrevocable (subject to the termination provision
below) license under any rights in any patent claims owned by
Facebook, to make, have made, use, sell, offer to sell, import, and
otherwise transfer the Software. For avoidance of doubt, no license
is granted under Facebook's rights in any patent claims that are
infringed by (i) modifications to the Software made by you or a third
party, or (ii) the Software in combination with any software or other
technology provided by you or a third party.

The license granted hereunder will terminate, automatically and
without notice, for anyone that makes any claim (including by filing
any lawsuit, assertion, or other action) alleging (a) direct,
indirect, or contributory infringement or inducement to infringe any
patent: (i) by Facebook or any of its subsidiaries or affiliates,
whether or not such claim is related to the Software, (ii) by any
party if such claim arises in whole or in part from any software,
product or service of Facebook or any of its subsidiaries or
affiliates, whether or not such claim is related to the Software, or
(iii) by any party relating to the Software; or (b) that any right in
any patent claim of Facebook is invalid or unenforceable.
"""

import numpy as np
import torch


def my_lu(A, permute_l=False):
    if len(A.shape) == 2:
        A = A.unsqueeze(0)
    M, N = A.shape[1], A.shape[2]
    K = int(np.min((M, N)))
    if A.shape[1] != A.shape[2]:
        m = int(np.max((A.shape[1], A.shape[2])))
        ten = torch.eye(m).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A)
        ten[:, : A.shape[1], :A.shape[2]] = A
        A = ten
    A_LU, pivots = torch.btrifact(A)
    P, A_L, A_U = torch.btriunpack(A_LU, pivots)
    P, A_L, A_U = P[:, :M, :M].squeeze(), A_L[:, :M, :K].squeeze(), A_U[:, :K, :N].squeeze()
    if permute_l:
        return torch.matmul(P, A_L), A_U
    else:
        return P, A_L, A_U

