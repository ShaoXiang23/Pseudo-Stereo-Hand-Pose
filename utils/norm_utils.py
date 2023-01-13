import torch

def putmask_gt(x, t1, t2):
    m = x.gt(t1)
    t2 = t2 * torch.ones_like(x).to(x.device)
    m_ = ~m
    return t2 * m + x * m_

def putmask_lt(x, t1, t2):
    m = x.lt(t1)
    t2 = t2 * torch.ones_like(x).to(x.device)
    m_ = ~m
    return t2 * m + x * m_

def putmask_ge(x, t1, t2):
    m = x.ge(t1)
    t2 = t2 * torch.ones_like(x).to(x.device)
    m_ = ~m
    return t2 * m + x * m_

def putmask_le(x, t1, t2):
    m = x.le(t1)
    t2 = t2 * torch.ones_like(x).to(x.device)
    m_ = ~m
    return t2 * m + x * m_

def norm_dep_img(dep, joint_z):
    lower_bound = torch.min(joint_z) - 0.05
    upper_bound = torch.max(joint_z) + 0.05

    dep = putmask_le(dep, lower_bound, upper_bound)
    min_dep = torch.min(dep) - 1e-3
    dep = putmask_ge(dep, upper_bound, 0.0)
    max_dep = torch.max(dep) + 1e-3
    dep = putmask_le(dep, min_dep, max_dep)
    range_dep = max_dep - min_dep
    dep = (-1 * dep + max_dep) / range_dep

    return dep

def batch_norm_dep_img(deps, joints):
    dep_norms = torch.ones_like(deps).to(deps.device)

    bs = deps.shape[0]
    for i in range(bs):
        joint_z = joints[i, :, -1]
        dep_norm = norm_dep_img(deps[i], joint_z)
        dep_norms[i] = dep_norm.to(deps.device)

    return dep_norms