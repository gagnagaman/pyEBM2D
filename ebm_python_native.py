import numpy as np
import netCDF4 as nc
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import os
import time
import matplotlib.pyplot as plt

def run_ebm():
    print("*" * 62)
    print("*****" + " " * 52 + "*****")
    print("*****             2D EBM PALEOCLIMATE MODEL              *****")
    print("*****            (Native Python Translation)             *****")
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

    # 2. Read Input Data (Geography & Albedo)
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../EBM/input'))
    
    geo_ds = nc.Dataset(os.path.join(input_dir, 'geography.nc'))
    geography = geo_ds.variables['landmask'][:]
    if geography.shape == (128, 65): geography = geography.T
    geo_ds.close()
    print(f" SUCCESSFULLY reading geography from [{input_dir}/geography.nc] into simulation!")
    
    albedo_ds = nc.Dataset(os.path.join(input_dir, 'albedo.nc'))
    albedo = albedo_ds.variables['ALBEDO'][:] 
    Pcoalbedo = 1.0 - albedo[:, 0, :, :] 
    albedo_ds.close()
    print(f" SUCCESSFULLY reading albedo from [{input_dir}/albedo.nc] into simulation!")

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
            
    # 5. Diffusion Coefficients & Operators
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

    # 6. Sparse Matrix Construction (Implicit Backward-Euler-like formulation)
    N = NX * NY
    def get_idx(j, i): return j * NX + i
    rows, cols, data = [], [], []
    
    # Poles equations
    for i in range(NX):
        if i == 0:
            idx = get_idx(0, i)
            rows.append(idx); cols.append(idx); data.append(GCnp)
            for k in range(NX):
                rows.append(idx); cols.append(get_idx(1, k)); data.append(-geom * DT2np[k])
                
            idx = get_idx(NY-1, i)
            rows.append(idx); cols.append(idx); data.append(GCsp)
            for k in range(NX):
                rows.append(idx); cols.append(get_idx(NY-2, k)); data.append(-geom * DT2sp[k])
        else:
            # Bind all pole points to the 0th point
            idx = get_idx(0, i)
            rows.append(idx); cols.append(idx); data.append(1.0)
            rows.append(idx); cols.append(get_idx(0, 0)); data.append(-1.0)
            
            idx = get_idx(NY-1, i)
            rows.append(idx); cols.append(idx); data.append(1.0)
            rows.append(idx); cols.append(get_idx(NY-1, 0)); data.append(-1.0)

    # Interior equations
    for j in range(1, NY-1):
        for i in range(NX):
            idx = get_idx(j, i)
            rows.append(idx); cols.append(idx); data.append(c0[j, i])
            rows.append(idx); cols.append(get_idx(j, (i-1)%NX)); data.append(-c1[j, i])
            rows.append(idx); cols.append(get_idx(j, (i+1)%NX)); data.append(-c3[j, i])
            
            if j - 1 == 0: rows.append(idx); cols.append(get_idx(0, 0)); data.append(-c2[j, i])
            else: rows.append(idx); cols.append(get_idx(j-1, i)); data.append(-c2[j, i])
                
            if j + 1 == NY - 1: rows.append(idx); cols.append(get_idx(NY-1, 0)); data.append(-c4[j, i])
            else: rows.append(idx); cols.append(get_idx(j+1, i)); data.append(-c4[j, i])
                
    A_mat = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Pre-factorize for instantaneous time steps. 
    # Convert to CSC format first to avoid efficiency warnings during factorize.
    solver = spla.factorized(A_mat.tocsc())

    # 7. Time Integration
    print("\n Start Energy Balance Model simulation:\n")
    print("  Year   Global Temperature")
    
    Temp = np.full((NY, NX), 5.0)
    LastRHS = np.zeros((NY, NX))
    Maxyrs = 100
    
    for year in range(Maxyrs):
        GTemp = 0.0
        for ts in range(NT):
            ts_for = ts
            ts_prev = (ts - 1) % NT
            F_cur = SF[ts_for]
            F_prev = SF[ts_prev]
            if ts == 0 and year == 0:
                F_prev = SF[-1] 
                
            # Right hand side follows Fortran implicit Crank-Nicolson-style scheme
            RHS_field = -(4.0 * heatcap * Temp / dt + F_cur + F_prev + LastRHS)
            
            b = np.zeros(N)
            for j in range(1, NY-1):
                for i in range(NX):
                    b[get_idx(j, i)] = -hh * RHS_field[j, i]
            b[get_idx(0, 0)] = -RHS_field[0, 0]
            b[get_idx(NY-1, 0)] = -RHS_field[NY-1, 0]
            
            x = solver(b)
            Temp = x.reshape((NY, NX))
            
            # Broadcast pole solution
            Temp[0, :] = Temp[0, 0]
            Temp[NY-1, :] = Temp[NY-1, 0]
            
            LastRHS = RHS_field
            
            ts_mean = 0.0
            for j in range(1, NY-1):
                for i in range(NX):
                    ts_mean += area[j] * Temp[j, i]
            ts_mean += area[0] * Temp[0, 0] + area[NY-1] * Temp[NY-1, 0]
            GTemp += ts_mean
            
        T_mean = GTemp / NT
        
        print(f"  {year+1:4d}    {T_mean:.5f}")
        
    print(f"\nEQUILIBRIUM REACHED AFTER {Maxyrs} YEARS. GLOBAL TEMP= {T_mean:.4f}")
    
    elapsed_time = (time.time() - start_time) / 60.0
    print(f"\n Elapsed time:      {elapsed_time:.2f} minutes")
    
    # Optionally save output to NetCDF to perfectly mirror Fortran output

    # Plotting
    lond = np.linspace(0, 360, NX, endpoint=False)
    plt.figure(figsize=(10, 5))
    plt.contourf(lond, latd, Temp, levels=20, cmap='inferno')
    plt.colorbar(label='Temperature (C)')
    plt.title('Native Python EBM Final Temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
    if not os.path.exists(fig_dir): fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, 'native_temperature.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, 'final_temperature.npy'), Temp)
    print(f" Final temperature results stored in {output_dir}/final_temperature.npy")

if __name__ == '__main__':
    run_ebm()
