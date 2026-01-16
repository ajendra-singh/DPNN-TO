import torch

def energy_vol(u_bc, xc, yc, zc, disp_u, disp_v, disp_w, dens, E_min, E_max, penal, nu, domain_volume):
    # Displacement field
    ux = u_bc + xc * disp_u(xc, yc, zc)
    uy = u_bc + xc * disp_v(xc, yc, zc)
    uz = u_bc + xc * disp_w(xc, yc, zc)

    # SIMP material interpolation
    young = E_min + (E_max - E_min) * dens.pow(penal)
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = young / (2 * (1 + nu))

    # Strains
    grad = lambda f: torch.autograd.grad(f.sum(), (xc, yc, zc), create_graph=True)
    ux_x, ux_y, ux_z = grad(ux)
    uy_x, uy_y, uy_z = grad(uy)
    uz_x, uz_y, uz_z = grad(uz)

    eps_xx = ux_x
    eps_yy = uy_y
    eps_zz = uz_z

    eps_xy = 0.5 * (ux_y + uy_x)
    eps_xz = 0.5 * (ux_z + uz_x)
    eps_yz = 0.5 * (uy_z + uz_y)

    # Stresses
    trace_eps = eps_xx + eps_yy + eps_zz

    sig_xx = lam * trace_eps + 2 * mu * eps_xx
    sig_yy = lam * trace_eps + 2 * mu * eps_yy
    sig_zz = lam * trace_eps + 2 * mu * eps_zz
    sig_xy = 2 * mu * eps_xy
    sig_xz = 2 * mu * eps_xz
    sig_yz = 2 * mu * eps_yz

    strain_energy = 0.5 * (eps_xx * sig_xx + eps_yy * sig_yy + eps_zz * sig_zz + 2.0 *(eps_xy * sig_xy + eps_xz * sig_xz + eps_yz * sig_yz))
    dV = domain_volume / xc.numel()
    # volume_frac = torch.sum(dens) * dA / domain_area
    internal_energy = torch.sum(strain_energy) * dV    
    return internal_energy #, volume_frac


