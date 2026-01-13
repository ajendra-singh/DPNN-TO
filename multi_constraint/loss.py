import torch

def energy_vol(u_bc, xc, yc, disp_u, disp_v, dens, E_min, E_max, penal, nu, domain_area):
    # Displacement field
    ux = u_bc + (1-yc) * disp_u(xc, yc)
    uy = u_bc + (1-yc) * disp_v(xc, yc)

    # SIMP material interpolation
    young = E_min + (E_max - E_min) * dens.pow(penal)
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = young / (2 * (1 + nu))

    # Strains
    grad = lambda f: torch.autograd.grad(f.sum(), (xc, yc), create_graph=True)
    ux_x, ux_y = grad(ux)
    uy_x, uy_y = grad(uy)
    eps_xx = ux_x
    eps_yy = uy_y
    eps_xy = 0.5 * (ux_y + uy_x)

    # Stresses
    sig_xx = (lam + 2 * mu) * eps_xx + lam * eps_yy
    sig_yy = (lam + 2 * mu) * eps_yy + lam * eps_xx
    sig_xy = 2 * mu * eps_xy

    vms = (sig_xx**2+sig_yy**2-sig_xx*sig_yy+3*sig_xy**2)**0.5

    # Multi-p-norm stress aggregation
    pnorm_str_4 = (torch.sum(vms**4) * (domain_area / xc.shape[0]))**(1/4)
    pnorm_str_6 = (torch.sum(vms**6) * (domain_area / xc.shape[0]))**(1/6)
    pnorm_str_8 = (torch.sum(vms**8) * (domain_area / xc.shape[0]))**(1/8)

    # Aggregate different p-norms (weighted sum or max)
    pnorm_str = (0.5*pnorm_str_4 + 0.3*pnorm_str_6 + 0.2*pnorm_str_8)

    # Energy density and integrals
    strain_energy = 0.5 * (eps_xx * sig_xx + eps_yy * sig_yy + 2 * eps_xy * sig_xy)
    dA = domain_area / xc.numel()

    volume_frac = torch.sum(dens) * dA / domain_area
    internal_energy = torch.sum(strain_energy) * dA
    
    return internal_energy, volume_frac, pnorm_str