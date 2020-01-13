# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


def compute_ap(scores, labels, npos, device=None):
    if device is None:
        device = scores.device

    if len(scores) == 0:
        return 0.0
    tp = labels == 1
    fp = labels == 0
    sc = scores
    assert tp.size() == sc.size()
    assert tp.size() == fp.size()
    sc, ind = torch.sort(sc, descending=True)
    tp = tp[ind].to(dtype=torch.float32)
    fp = fp[ind].to(dtype=torch.float32)
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)

    # # Compute precision/recall
    rec = tp / npos
    prec = tp / (fp + tp)
    ap = xVOCap(rec, prec, device)

    return ap


def xVOCap(rec, prec, device):

    z = rec.new_zeros((1))
    o = rec.new_ones((1))
    mrec = torch.cat((z, rec, o))
    mpre = torch.cat((z, prec, z))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = (mrec[1:] != mrec[0:-1]).nonzero()[:, 0] + 1
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap
