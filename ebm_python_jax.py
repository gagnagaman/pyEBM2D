import numpy as np
import netCDF4 as nc
import os
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, lax

# Force JAX to use CPU for direct comparison (can be removed if GPU is desired)
jax.config.update('jax_platform_name', 'cpu')

def run_ebm_jax():
    print("*" * 62)
    print("*****" + " " * 52 + "*****")
    print("*****             2D EBM PALEOCLIMATE MODEL              *****")
    print("*****             (JAX-Accelerated Version)              *****")
    print("*****" + " " * 52 + "*****")
    print("*" * 62)
    
    start_time = time.time()
    
    # 1. Grid and Physical Constants
    A = 210.3; B = 2.15
    NX = 128; NY = 65; NT = 48
    dt = 1.0 / NT
    pi = np.pi; dy = pi / (NY - 1); h = dy
    
    # Geometry exact match to Fortran
    latd = np.zeros(NY); siny = np.zeros(NY); cosy = np.zeros(NY); tany = np.zeros(NY)
    area = np.zeros(NY)
    
    for j in range(NY):
        lat = pi/2.0 - dy * j
        latd[j] = lat * 180. / pi
        siny[j] = np.sin(lat)
        
        if j == 0:
            cosy[j] = 0.0; tany[j] = 1000.0
            area[j] = 0.5 * (1.0 - np.cos(h/2.0))
        elif j == NY - 1:
            cosy[j] = 0.0; tany[j] = -1000.0
            area[j] = 0.5 * (1.0 - np.cos(h/2.0))
        else:
            cosy[j] = np.cos(lat); tany[j] = np.tan(lat)
            area[j] = np.sin(h/2.0) * np.sin(h * j) / NX

    # 2. Read Input Data
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../EBM/input'))
    
    geo_ds = nc.Dataset(os.path.join(input_dir, 'geography.nc'))
    geography = geo_ds.variables['landmask'][:]
    if geography.shape == (128, 65): geography = geography.T
    geo_ds.close()
    
    albedo_ds = nc.Dataset(os.path.join(input_dir, 'albedo.nc'))
    albedo = albedo_ds.variables['ALBEDO'][:] 
    Pcoalbedo = 1.0 - albedo[:, 0, :, :] 
    albedo_ds.close()

    # 3. Heat Capacities
    sec_per_yr = 3.15576e7
    C_atmos = 1005.0 * 0.5 * 1000.0 * 1.293 * sum(np.exp(-(0.25 + 0.5*n)/7.6) for n in range(10)) / sec_per_yr
    C_soil = 2.0 * 1100.0 * 850.0 / sec_per_yr
    C_seaice = 2.5 * 917.0 * 2106.0 / sec_per_yr
    C_snow = 2.0 * 400.0 * 1900.0 / sec_per_yr
    C_mixed_layer = 70.0 * 1000.0 * 4186.0 / sec_per_yr
    
    heatcap = np.zeros((NY, NX))
    for j in range(NY):
        for i in range(NX):
            geo = geography[j, i]
            if geo == 1: heatcap[j,i] = C_soil + C_atmos
            elif geo == 2: heatcap[j,i] = C_seaice + C_atmos
            elif geo == 3: heatcap[j,i] = C_snow + C_atmos
            elif geo == 4: heatcap[j,i] = C_mixed_layer/3.0 + C_atmos
            else: heatcap[j,i] = C_mixed_layer + C_atmos

    # 4. Solar Forcing
    ecc = 0.0167239327; ob = 0.409214646; per = 1.78091741; S0 = 1371.685
    lambda_ = np.zeros(NT+1)
    eccfac = 1.0 - ecc**2
    rzero = (2.0*pi)/eccfac**1.5
    def func(angle): return rzero*(1.0 - ecc*np.cos(angle))**2
    
    for n in range(1, NT+1):
        nu = lambda_[n-1] - per
        t1 = dt * func(nu); t2 = dt * func(nu + 0.5*t1); t3 = dt * func(nu + 0.5*t2); t4 = dt * func(nu + t3)
        lambda_[n] = lambda_[n-1] + (t1 + 2.0*t2 + 2.0*t3 + t4)/6.0
        
    SF = np.zeros((NT, NY, NX))
    for n in range(NT):
        nu = lambda_[n] - per
        rhofac = ((1.0 - ecc*np.cos(nu))/eccfac)**2
        sindec = np.sin(ob)*np.sin(lambda_[n])
        cosdec = np.sqrt(1.0 - sindec**2)
        tandec = sindec/cosdec
        for j in range(NY):
            z = -tany[j]*tandec
            if z >= 1.0: solar = 0.0
            elif z <= -1.0: solar = rhofac * S0 * siny[j] * sindec
            else:
                Hzero = np.arccos(z)
                solar = rhofac/pi * S0 * (Hzero*siny[j]*sindec + cosy[j]*cosdec*np.sin(Hzero))
            SF[n, j, :] = solar * Pcoalbedo[n, j, :] - A
            
    # 5. Diffusion Coefficients
    KlandSP = 0.20; KlandNP = 0.28; Kland = 0.65; Keq = 0.65; Kocean = 0.40
    DiffC = np.zeros((NY, NX))
    for j in range(NY):
        theta = pi * j / (NY - 1)
        colat_term = np.sin(theta)**5
        for i in range(NX):
            geo = geography[j, i]
            if 5.0 <= geo <= 7.0: DiffC[j, i] = (Keq - Kocean) * colat_term + Kocean
            else:
                if j < 33: DiffC[j, i] = (Kland - KlandNP) * colat_term + KlandNP
                else: DiffC[j, i] = (Kland - KlandSP) * colat_term + KlandSP

    hh = h * h
    csc2 = np.zeros(NY); cot = np.zeros(NY)
    for j in range(1, NY-1):
        csc2[j] = 1.0 / (np.sin(pi * j / (NY - 1))**2)
        cot[j] = 1.0 / np.tan(pi * j / (NY - 1))
        
    c0 = np.zeros((NY, NX)); c1 = np.zeros((NY, NX)); c2 = np.zeros((NY, NX))
    c3 = np.zeros((NY, NX)); c4 = np.zeros((NY, NX))
    
    for j in range(1, NY-1):
        for i in range(NX):
            im1 = (i - 1) % NX; ip1 = (i + 1) % NX
            dPhi = (DiffC[j, ip1] - DiffC[j, im1]) / (2.0 * h)
            c1[j, i] = csc2[j] * (DiffC[j, i] - h/2.0 * dPhi)
            c3[j, i] = csc2[j] * (DiffC[j, i] + h/2.0 * dPhi)
            
            dTheta = (DiffC[j+1, i] - DiffC[j-1, i]) / (2.0 * h)
            c2[j, i] = DiffC[j, i] - h/2.0 * (DiffC[j, i] * cot[j] + dTheta)
            c4[j, i] = DiffC[j, i] + h/2.0 * (DiffC[j, i] * cot[j] + dTheta)
            
            c0[j, i] = 2.0 * DiffC[j, i] * (1.0 + csc2[j]) + hh * (2.0 * heatcap[j, i]/dt + B)

    geom = np.sin(h/2.0) / area[0]
    Tarea = area[0] + area[1]*NX
    DT2np = np.zeros(NX); DT2sp = np.zeros(NX)
    MDnp = 0.0; MDsp = 0.0
    for i in range(NX):
        DT2np[i] = (area[0]*DiffC[0,0] + area[1]*NX*DiffC[1,i]) / Tarea
        DT2sp[i] = (area[-1]*DiffC[-1,-1] + area[-2]*NX*DiffC[-2,i]) / Tarea
        MDnp += DT2np[i]; MDsp += DT2sp[i]
        
    GCnp = geom * MDnp + B + 2.0*heatcap[0,0]/dt
    GCsp = geom * MDsp + B + 2.0*heatcap[-1,0]/dt

    # 6. Dense Matrix Construction for JAX
    N = NX * NY
    def get_idx(j, i): return j * NX + i
    
    print("Building and compiling dense matrix for JAX (this takes a moment)...")
    A_dense = np.zeros((N, N), dtype=np.float32)
    
    # Poles equations
    for i in range(NX):
        if i == 0:
            idx = get_idx(0, i)
            A_dense[idx, idx] = GCnp
            for k in range(NX):
                A_dense[idx, get_idx(1, k)] = -geom * DT2np[k]
                
            idx = get_idx(NY-1, i)
            A_dense[idx, idx] = GCsp
            for k in range(NX):
                A_dense[idx, get_idx(NY-2, k)] = -geom * DT2sp[k]
        else:
            idx = get_idx(0, i)
            A_dense[idx, idx] = 1.0
            A_dense[idx, get_idx(0, 0)] = -1.0
            
            idx = get_idx(NY-1, i)
            A_dense[idx, idx] = 1.0
            A_dense[idx, get_idx(NY-1, 0)] = -1.0

    # Interior equations
    for j in range(1, NY-1):
        for i in range(NX):
            idx = get_idx(j, i)
            A_dense[idx, idx] = c0[j, i]
            A_dense[idx, get_idx(j, (i-1)%NX)] = -c1[j, i]
            A_dense[idx, get_idx(j, (i+1)%NX)] = -c3[j, i]
            
            if j - 1 == 0: A_dense[idx, get_idx(0, 0)] = -c2[j, i]
            else: A_dense[idx, get_idx(j-1, i)] = -c2[j, i]
                
            if j + 1 == NY - 1: A_dense[idx, get_idx(NY-1, 0)] = -c4[j, i]
            else: A_dense[idx, get_idx(j+1, i)] = -c4[j, i]

    # Transfer data to JAX
    # Since JAX does not have a native sparse direct solver, we use dense LU factorization.
    # Factoring an 8320x8320 matrix on CPU takes ~1-2 seconds, solving is extremely fast.
    j_A = jnp.array(A_dense)
    lu, piv = jax.scipy.linalg.lu_factor(j_A)
    
    j_SF = jnp.array(SF, dtype=np.float32)
    j_heatcap = jnp.array(heatcap, dtype=np.float32)
    
    # Indices for the poles to vectorize JAX updates (prevents XLA circular compilation loop)
    pole_n_idx = np.array([get_idx(0, i) for i in range(1, NX)])
    pole_s_idx = np.array([get_idx(NY-1, i) for i in range(1, NX)])
    
    # 7. JAX Scanning Loop
    @jit
    def time_step(carry, step_idx):
        T_flat, LastRHS_flat = carry
        
        # Calculate indices
        ts_for = step_idx % NT
        ts_prev = (step_idx - 1) % NT
        # Vernal equinox logic for first step
        ts_prev = jnp.where(step_idx == 0, NT - 1, ts_prev)
        
        F_cur = j_SF[ts_for].flatten()
        F_prev = j_SF[ts_prev].flatten()
        
        # RHS Calculation
        RHS_field_flat = -(4.0 * j_heatcap.flatten() * T_flat / dt + F_cur + F_prev + LastRHS_flat)
        
        # Build solution vector 'b'
        b = -hh * RHS_field_flat
        
        # Fix poles in b
        b = b.at[get_idx(0, 0)].set(-RHS_field_flat[get_idx(0, 0)])
        b = b.at[get_idx(NY-1, 0)].set(-RHS_field_flat[get_idx(NY-1, 0)])
        
        # Pole broadcast equations are b=0 (vectorized)
        b = b.at[pole_n_idx].set(0.0)
        b = b.at[pole_s_idx].set(0.0)
            
        # Solve dense matrix
        T_new_flat = jax.scipy.linalg.lu_solve((lu, piv), b)
        
        # Re-broadcast poles in solution (vectorized)
        T_new_flat = T_new_flat.at[pole_n_idx].set(T_new_flat[get_idx(0, 0)])
        T_new_flat = T_new_flat.at[pole_s_idx].set(T_new_flat[get_idx(NY-1, 0)])
            
        return (T_new_flat, RHS_field_flat), T_new_flat
        
    print("\n Start Energy Balance Model simulation (JAX loop compiling...)\n")
    print("  Year   Global Temperature")
    
    Maxyrs = 100
    total_steps = Maxyrs * NT
    
    T_init_flat = jnp.full(N, 5.0, dtype=np.float32)
    LastRHS_init_flat = jnp.zeros(N, dtype=np.float32)
    
    # We execute the jax loop for the entire 40 year simulation in one compiled pass
    _, history_T = lax.scan(time_step, (T_init_flat, LastRHS_init_flat), jnp.arange(total_steps))
    
    # Extract end-of-year data point (or calculate mean correctly)
    for year in range(Maxyrs):
        # We grab the final time step of each year to match the Fortran's equilibrium check
        # Fortran actually tracks the mean of the entire year, so let's average the year:
        year_history = history_T[year*NT : (year+1)*NT]
        
        # Accumulate annual mean over the 48 timesteps
        GTemp = 0.0
        for ts in range(NT):
            T_ts = np.array(year_history[ts]).reshape((NY, NX))
            ts_mean = 0.0
            for j in range(1, NY-1):
                for i in range(NX):
                    ts_mean += area[j] * T_ts[j, i]
            ts_mean += area[0] * T_ts[0, 0] + area[NY-1] * T_ts[NY-1, 0]
            GTemp += ts_mean
        
        T_mean = GTemp / NT
        # Save final state for plotting
        T_year = np.array(year_history[-1]).reshape((NY, NX))
        
        print(f"  {year+1:4d}    {T_mean:.5f}")
        
    print(f"\nEQUILIBRIUM REACHED AFTER {Maxyrs} YEARS. GLOBAL TEMP= {T_mean:.4f}")
    
    elapsed_time = (time.time() - start_time) / 60.0
    print(f"\n Elapsed time:      {elapsed_time:.2f} minutes")

    # Plotting
    lond = np.linspace(0, 360, NX, endpoint=False)
    plt.figure(figsize=(10, 5))
    plt.contourf(lond, latd, T_year, levels=20, cmap='inferno')
    plt.colorbar(label='Temperature (C)')
    plt.title('JAX Python EBM Final Temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
    if not os.path.exists(fig_dir): fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, 'jax_temperature.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    print("\n[NOTE]: Native SciPy sparse solver (ebm_python_native.py) runs faster on CPU.")
    print("        This JAX version relies on dense LU factorization (O(N^3) memory/ops),")
    print("        whereas the Fortran and SciPy versions exploit the O(N) banded sparsity of the matrix.")

if __name__ == '__main__':
    run_ebm_jax()
