import torch


def SpKL(output, target, a2, beta=1, rho=0.05):
    quadratic = torch.pow(output - target, 2)
    rho_hat = torch.mean(a2, dim=0)
    rho = rho * torch.ones(a2.shape[1])
    KLs = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    loss = torch.sum(quadratic) / 0.5 + beta * torch.sum(KLs)
    return loss


def L2norm(output, target, theta, lbda=0.05):
    NL = theta.shape[1]
    quadratic = torch.pow(output - target, 2)
    L2 = theta.mm(theta.t())
    J = torch.sum(quadratic) / (2 * NL) + lbda * L2 / 2
    return J
