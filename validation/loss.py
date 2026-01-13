import torch

def energy_vol(u_bc, xc, yc, disp_u, disp_v, dens, E_min, E_max, penal, nu, domain_area):
    # Displacement field
    ux = u_bc + xc * disp_u(xc, yc)
    uy = u_bc + xc * disp_v(xc, yc)

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

    # Energy density and integrals
    strain_energy = 0.5 * (eps_xx * sig_xx + eps_yy * sig_yy + 2 * eps_xy * sig_xy)
    dA = domain_area / xc.numel()

    volume_frac = torch.sum(dens) * dA / domain_area
    internal_energy = torch.sum(strain_energy) * dA
    
    return internal_energy, volume_frac