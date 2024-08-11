import numpy as np
from fof_analysis import fof

def get_gas_star_properties(part):
    """
    Extract gas properties from the input data and organize them into a dictionary, keyed by snapshot.

    Parameters:
    part (dict): A dictionary containing gas particle data.
    snapshot (str): Snapshot identifier.

    Returns:
    dict: A dictionary containing gas properties keyed by snapshot.

    Example:
    >>> gas_props = get_gas_properties(part, 'snapshot_001')
    >>> print(gas_props['g_x_snapshot_001'])
    """
    return {
        'g_rxyz': part['gas'].prop('host.distance.total'),  # Total distance of gas particles
        'g_x': part['gas'].prop('host.distance.principal.cartesian')[:, 0],  # x-coordinates of gas particles
        'g_y': part['gas'].prop('host.distance.principal.cartesian')[:, 1],  # y-coordinates of gas particles
        'g_z': part['gas'].prop('host.distance.principal.cartesian')[:, 2],  # z-coordinates of gas particles
        'g_Rxy': part['gas'].prop('host.distance.principal.cylindrical')[:, 0],  # Cylindrical radial distance of gas particles
        'g_vR': part['gas'].prop('host.velocity.principal.cylindrical')[:, 0],  # Radial velocity of gas particles
        'g_vphi': part['gas'].prop('host.velocity.principal.cylindrical')[:, 2],  # Azimuthal velocity of gas particles
        'g_vz2': part['gas'].prop('host.velocity.principal.cylindrical')[:, 1],  # Vertical velocity of gas particles
        'g_vx':part['gas'].prop('host.velocity.principal.cartesian')[:, 0],  # x-component of velocity of gas particles
        'g_vy': part['gas'].prop('host.velocity.principal.cartesian')[:, 1],  # y-component of velocity of gas particles
        'g_vz': part['gas'].prop('host.velocity.principal.cartesian')[:, 2],  # z-component of velocity of gas particles
        'g_mass': part['gas'].prop('mass'),  # Mass of gas particles
        'g_size': part['gas'].prop('size'),  # Size of gas particles
        'g_feh': part['gas'].prop('metallicity.fe'),  # Iron metallicity of gas particles
        'g_ids': part['gas']['id'],  # IDs of gas particles

        'rxyz': part['star'].prop('host.distance.total'),  # Total distance of star particles
        'x': part['star'].prop('host.distance.principal.cartesian')[:, 0],  # x-coordinates of star particles
        'y': part['star'].prop('host.distance.principal.cartesian')[:, 1],  # y-coordinates of star particles
        'z': part['star'].prop('host.distance.principal.cartesian')[:, 2],  # z-coordinates of star particles
        'Rxy': part['star'].prop('host.distance.principal.cylindrical')[:, 0],  # Cylindrical radial distance of star particles
        'vR': part['star'].prop('host.velocity.principal.cylindrical')[:, 0],  # Radial velocity of star particles
        'vphi': part['star'].prop('host.velocity.principal.cylindrical')[:, 2],  # Azimuthal velocity of star particles
        'vz2': part['star'].prop('host.velocity.principal.cylindrical')[:, 1],  # Vertical velocity of star particles
        'vx': part['star'].prop('host.velocity.principal.cartesian')[:, 0],  # x-component of velocity of star particles
        'vy': part['star'].prop('host.velocity.principal.cartesian')[:, 1],  # y-component of velocity of star particles
        'vz': part['star'].prop('host.velocity.principal.cartesian')[:, 2],  # z-component of velocity of star particles
        'age': part['star'].prop('age'),  # Age of star particles
        'mass':part['star'].prop('mass'),  # Mass of star particles
        'feh': part['star'].prop('metallicity.fe'),  # Iron metallicity of star particles
        'ids':  part['star']['id'],  # IDs of star particles
        'id_generation': part['star']['id.generation'],  # Generation ID of star particles
        'id_child': part['star']['id.child']  # Child ID of star particles
        
    }
    
def process_first_snapshot(gs_props, age_cut_Myr=2, age_cut_Min_Myr=1, cluster_select = 0, b_pc=4, ncut_min=5):

    # Convert age cuts to Gyr
    age_cut_Gyr = age_cut_Myr / 1e3
    age_cut_Min_Gyr = age_cut_Min_Myr / 1e3

    # Convert binding length to kpc
    b_kpc = b_pc / 1e3

    # Indices of stars we want to run fof on (to find star clusters)
    si = np.where((gs_props['age'] <= age_cut_Gyr) & (gs_props['age'] >= age_cut_Min_Gyr) & (gs_props['rxyz'] < 20) & (abs(gs_props['z']) < 1.5))
     #running fof
    ind, sxcm, sycm, szcm, mtot, grpid, r90, r50, srmax =fof.find(gs_props['x'][si], gs_props['y'][si], gs_props['z'][si], b=b_kpc, mass=gs_props['mass'][si], ncut=ncut_min)

   
    # Selecting just arrays associated with young stars that fof was run on
    idsyoungstars            = gs_props['ids'][si]
    id_generation_youngstars = gs_props['id_generation'][si]
    id_child_youngstars      = gs_props['id_child'][si]
    x_youngstars             = gs_props['x'][si]
    y_youngstars             = gs_props['y'][si]
    z_youngstars             = gs_props['z'][si]

    #Saving information for the first cluster
    ids_cluster            = idsyoungstars[ind[cluster_select]]  
    id_generation_cluster  = id_generation_youngstars[ind[cluster_select]]
    id_child_cluster       = id_child_youngstars[ind[cluster_select]]
    x_cluster              = x_youngstars[ind[cluster_select]]
    y_cluster              = y_youngstars[ind[cluster_select]]
    z_cluster              = z_youngstars[ind[cluster_select]]
   
    return {
        'ids_cluster': ids_cluster,
        'id_generation_cluster': id_generation_cluster,
        'id_child_cluster': id_child_cluster,
        'x_cluster': x_cluster,
        'y_cluster': y_cluster,
        'z_cluster': z_cluster,
        'ind' : ind,
        'sxcm' : sxcm,
        'sycm' : sycm,
        'szcm' : szcm
        
    }
def preserve_ids(ids_cluster, id_generation_cluster, id_child_cluster, ids, id_generation, id_child):
    """
    Preserves indices of elements in ids_cluster that match specified IDs, generations, and child IDs.

    Parameters:
    ids_cluster (np.array): Array of IDs to be preserved.
    id_generation_cluster (np.array): Array of generation IDs corresponding to ids_cluster.
    id_child_cluster (np.array): Array of child IDs corresponding to ids_cluster.
    ids (np.array): Array of all IDs.
    id_generation (np.array): Array of all generation IDs.
    id_child (np.ndarray): Array of all child IDs.

    Returns:
    s_loc: Array of indices pointing to matching elements in ids, id_generation, and id_child for each element in ids_cluster.

    Example:
    >>> s_loc_indices = preserve_ids(ids_cluster_array, id_generation_cluster_array, id_child_cluster_array, ids_array, id_generation_array, id_child_array)
    >>> print(s_loc_indices)
    """
    s_loc = np.zeros(len(ids_cluster), dtype=int)  # Initialize s_loc as a 1-dimensional numpy array of zeros
    
    # Loop over each element in ids_cluster
    for i in range(len(ids_cluster)):
        input_id = ids_cluster[i]  # Current ID from ids_cluster
        input_generation = id_generation_cluster[i]  # Current generation from id_generation_cluster
        input_child = id_child_cluster[i]  # Current child ID from id_child_cluster
        
        # Find indices where input_id, input_generation, and input_child match corresponding values in ids, id_generation, and id_child
        keep = np.where((input_id == ids) & (input_generation == id_generation) & (input_child == id_child))
        
        if keep[0].size > 0:  # If there's at least one matching element
            s_loc[i] = keep[0][0]  # Store the index of the first matching element in s_loc

    return s_loc  # Return the array s_loc containing indices of matching elements

def find_new_snap_info(s_loc, gas_star_properties_data):
    """
    Extracts cluster information based on the specified location indices and organizes them into a dictionary.

    Parameters:
    s_loc (np.ndarray): Array of indices indicating the location of the selected cluster.
    ids (np.ndarray): Array of IDs for all particles.
    x (np.ndarray): Array of x-coordinates for all particles.
    y (np.ndarray): Array of y-coordinates for all particles.
    z (np.ndarray): Array of z-coordinates for all particles.
    mass (np.ndarray): Array of masses for all particles.
    snapshot (str): Snapshot identifier.

    Returns:
    dict: A dictionary containing snapshot-specific cluster information.

    Example:
    >>> snapshot_data = find_new_snap_info(s_loc_array, ids_array, x_coords_array, y_coords_array, z_coords_array, mass_array, '001')
    >>> print(snapshot_data['ids_cluster_1_001'])
    """
    x    = gas_star_properties_data['x']
    y    = gas_star_properties_data['y']
    z    = gas_star_properties_data['z']
    mass = gas_star_properties_data['mass']
    ids =  gas_star_properties_data['ids']
    
    # Extract cluster information using the specified location indices
    ids_cluster = ids[s_loc]      # IDs of the selected cluster
    x_cluster = x[s_loc]          # x-coordinates of the selected cluster
    y_cluster = y[s_loc]          # y-coordinates of the selected cluster
    z_cluster = z[s_loc]          # z-coordinates of the selected cluster
    mass_cluster = mass[s_loc]    # Masses of the selected cluster

    # Create a dictionary with the snapshot-specific cluster information
    cluster_info = {
        'ids_cluster': ids_cluster,      # IDs of the cluster
        'x_cluster': x_cluster,          # x-coordinates of the cluster
        'y_cluster': y_cluster,          # y-coordinates of the cluster
        'z_cluster': z_cluster,          # z-coordinates of the cluster
        'mass_cluster': mass_cluster     # Masses of the cluster
    }

    # Return the dictionary containing cluster information
    return cluster_info
    
def get_gas_star_indices(xcm1, ycm1, zcm1, gas_star_properties_data, pc_width=25):
    """
    Calculate indices of gas particles within a cubic region centered on the cluster center.

    Parameters:
    xcm1 (float): X-coordinate of the cluster center.
    ycm1 (float): Y-coordinate of the cluster center.
    zcm1 (float): Z-coordinate of the cluster center.
    g_x (np.ndarray): Array of x-coordinates of gas particles.
    g_y (np.ndarray): Array of y-coordinates of gas particles.
    g_z (np.ndarray): Array of z-coordinates of gas particles.
    snapshot (str): Snapshot identifier.
    pc_width (float, optional): Width of the cubic region in parsecs (default is 25 pc).

    Returns:
    dict: A dictionary containing indices of gas particles within the region, keyed by snapshot.
    
    Example:
    >>> g_indices = get_gas_indices(0.0, 0.0, 0.0, g_x, g_y, g_z, '001')
    >>> print(g_indices['g_ind_001'])
    """
    g_x = gas_star_properties_data['g_x']
    g_y = gas_star_properties_data['g_y']
    g_z= gas_star_properties_data['g_z']
    x = gas_star_properties_data['x']
    y = gas_star_properties_data['y'] 
    z = gas_star_properties_data['z']
    
    # Convert parsec width to kiloparsecs
    kpc_width = pc_width / 1e3

    # Find indices of gas particles within the specified cubic region

    # Create a dictionary with snapshot-specific gas indices
    #print('xcm1 + kpc_width:', xcm1 - kpc_width, xcm1 + kpc_width )
    #print('ycm1 + kpc_width:', ycm1 - kpc_width, ycm1 + kpc_width )
    
    g_ind =  np.where((g_x < (xcm1 + kpc_width)) &  # Condition for x-coordinates
                     (g_x > (xcm1 - kpc_width)) &
                     (g_y < (ycm1 + kpc_width)) &  # Condition for y-coordinates
                     (g_y > (ycm1 - kpc_width)) &
                     (g_z < (zcm1 + kpc_width)) &  # Condition for z-coordinates
                     (g_z > (zcm1 - kpc_width)))
    
    # Find indices of star particles within the specified cubic region
   

    # Create a dictionary with snapshot-specific star indices
    s_ind =  np.where((x < (xcm1 + kpc_width)) &  # Condition for x-coordinates
                     (x > (xcm1 - kpc_width)) &
                     (y < (ycm1 + kpc_width)) &  # Condition for y-coordinates
                     (y > (ycm1 - kpc_width)) &
                     (z < (zcm1 + kpc_width)) &  # Condition for z-coordinates
                     (z > (zcm1 - kpc_width)))
    #print("gind", g_ind[0][0], g_y[g_ind[0][0]])
    return g_ind[0], s_ind[0] 

def gas_star_coordinates(gas_star_properties, g_ind, x_cluster, y_cluster, z_cluster):
    """
    Extract coordinates and properties of gas and star particles, and organize them into a dictionary.

    Parameters:
    g_x (np.ndarray): Array of x-coordinates of gas particles.
    g_y (np.ndarray): Array of y-coordinates of gas particles.
    g_z (np.ndarray): Array of z-coordinates of gas particles.
    g_mass (np.ndarray): Array of masses of gas particles.
    g_size (np.ndarray): Array of sizes of gas particles.
    g_ind (np.ndarray): Indices of the gas particles of interest.
    x_cluster (np.ndarray): Array of x-coordinates of star particles in the cluster.
    y_cluster (np.ndarray): Array of y-coordinates of star particles in the cluster.
    z_cluster (np.ndarray): Array of z-coordinates of star particles in the cluster.
    snapshot (str): Snapshot identifier.

    Returns:
    dict: A dictionary containing coordinates and properties of gas and star particles, keyed by snapshot.
    
    Example:
    >>> gas_star_coords = gas_star_coordinates(g_x, g_y, g_z, g_mass, g_size, g_ind, x_cluster, y_cluster, z_cluster)
    >>> print(gas_star_coords['g_xs])
    """

    # Extract gas coordinates and properties based on the gas indices
    g_xs = gas_star_properties['g_x'][g_ind]      # x-coordinates of the gas particles
    g_ys = gas_star_properties['g_y'][g_ind]      # y-coordinates of the gas particles
    g_zs = gas_star_properties['g_z'][g_ind]      # z-coordinates of the gas particles
    g_mg = gas_star_properties['g_mass'][g_ind]   # Mass of the gas particles
    g_hg = gas_star_properties['g_size'][g_ind]   # Size of the gas particles

    # Use cluster coordinates directly for stars
    s_xs = x_cluster       # x-coordinates of the star particles
    s_ys = y_cluster       # y-coordinates of the star particles
    s_zs = z_cluster       # z-coordinates of the star particles

    # Create a dictionary with the gas and star coordinates
    coordinates = {
        'g_xs': g_xs,   # x-coordinates of the gas particles
        'g_ys': g_ys,   # y-coordinates of the gas particles
        'g_zs': g_zs,   # z-coordinates of the gas particles
        'g_mg': g_mg,   # Mass of the gas particles
        'g_hg': g_hg,   # Size of the gas particles
        's_xs': s_xs,   # x-coordinates of the star particles
        's_ys': s_ys,   # y-coordinates of the star particles
        's_zs': s_zs    # z-coordinates of the star particles
    }

    # Return the dictionary
    return coordinates
    
def compute_velocity_array_xy(xcen, ycen, zcen, coordinates, stepn = 16, pc_width = 40, nkernel = 200, kernel_file_path = '/home1/09528/amoac/'):
    g_x = coordinates['g_xs']
    g_y = coordinates['g_ys']
    g_z = coordinates['g_zs']
    g_mass = coordinates['g_mg']
    g_size = coordinates['g_hg']
    #print("g_y[0]", g_y[0])
    kpc_width = pc_width/1e3
    sizebox = 2*kpc_width
    #sizebox = pc_width
    res_xy = sizebox/stepn
    stepnh = (stepn-1)/2
    
    kernelfile = kernel_file_path + 'kernel2d'
    w2d_kernel = []
    w2d_table = np.zeros(nkernel+3)
    temp = open(kernelfile,'r').read().split('\n')
    

    for i in range(len(temp)-1):
        if (i%2 == 0):
            w2d_kernel.append(float(temp[i]))

    w2d_kernel.append(0.)
    w2d_kernel = np.array(w2d_kernel)

    
    w2darray = np.zeros(shape=(stepn, stepn))
    vel_array_xy = np.zeros(shape=(stepn, stepn))
    #ngas = len(ig[0])
    ngas = len(g_x)
    # Set the kernel weights, make sure to normalize properly
    htot = 0
    for i in range(ngas):
        htot += g_size[i] / ngas
    
       
    #print('First grid:', xcen - (stepnh*res), ycen + (stepnh*res),ycen - (stepnh*res))
    # Big loop
    for i in range(ngas):
        xi = (g_x[i] - xcen) / res_xy + stepnh
        yi = (g_y[i] - ycen) / res_xy + stepnh
        hi = 2 * g_size[i] / res_xy
        
        ixmin = int(np.floor(xi - hi)) + 1
        #print('x min',ixmin, xi, g_x[i])
        
        ixmin = max(ixmin, 0)
        ixmin = min(ixmin, stepn - 1)

        ixmax = int(np.floor(xi + hi))
        #print('x max',ixmax)
        ixmax = max(ixmax, 0)
        ixmax = min(ixmax, stepn - 1)

        iymin = int(np.floor(yi - hi)) + 1
        #print('y min',iymin, yi, g_y[i], 'y cen:',ycen, 'res and stepn',res,stepnh)
        iymin = max(iymin, 0)
        iymin = min(iymin, stepn - 1)

        iymax = int(np.floor(yi + hi))
        #print('y max',iymax)
        iymax = max(iymax, 0)
        iymax = min(iymax, stepn - 1)
        
        

        w2d_sum = 0.
        d2max = 4. * g_size[i] * g_size[i]
        kernfrac = nkernel * 0.5 / g_size[i]
        loopi = np.arange(ixmin, ixmax + 1, 1)
        loopj = np.arange(iymin, iymax + 1, 1)
        for ii in range(len(loopi)):
            r0xx = (ii - stepnh) * res_xy + xcen
            for jj in range(len(loopj)):
                r0yy = (jj - stepnh) * res_xy + ycen
                xx = g_x[i] - r0xx
                yy = g_y[i] - r0yy
                d2 = xx * xx + yy * yy
                if d2 <= d2max:
                    d = np.sqrt(d2)
                    xii = d * kernfrac
                    xxi = int(np.floor(xii))
                    w2d = (w2d_table[xxi + 1] - w2d_table[xxi]) * (xii - xxi) + w2d_table[xxi]
                    w2darray[ii, jj] = w2d
                    w2d_sum += w2d
                else:
                    w2darray[ii, jj] = 0.

        if w2d_sum > 0:
            weight = g_mass[i] / w2d_sum
            for ii in range(len(loopi)):
                for jj in range(len(loopj)):
                    if w2darray[ii, jj] > 0:
                        vel_array_xy[ii, jj] += w2darray[ii, jj] * weight
        else:
            ii = int(np.floor(xi + 0.5))
            jj = int(np.floor(yi + 0.5))
            if ((ii >=0) & (ii <= stepn-1) & (jj >= 0) & (jj <= stepn-1)):
                vel_array_xy[ii,jj] = vel_array_xy[ii,jj] + g_mass[i]
        
        i
    #return vel_array[0,:,:]
    extent_xy=(xcen - kpc_width, xcen + kpc_width, ycen - kpc_width, ycen + kpc_width)
    return vel_array_xy, res_xy, extent_xy

def compute_velocity_array_xz(xcen, ycen, zcen, coordinates, stepn = 16, pc_width = 40, nkernel = 200, kernel_file_path = '/home1/09528/amoac/'):
    g_x = coordinates['g_xs']
    g_z = coordinates['g_zs']
    g_mass = coordinates['g_mg']
    g_size = coordinates['g_hg']
    #print("g_y[0]", g_y[0])
    kpc_width = pc_width/1e3
    sizebox = 2*kpc_width
    #sizebox = pc_width
    res_xz = sizebox/stepn
    stepnh = (stepn-1)/2
    
    kernelfile = kernel_file_path + 'kernel2d'
    w2d_kernel = []
    w2d_table = np.zeros(nkernel+3)
    temp = open(kernelfile,'r').read().split('\n')
    

    for i in range(len(temp)-1):
        if (i%2 == 0):
            w2d_kernel.append(float(temp[i]))

    w2d_kernel.append(0.)
    w2d_kernel = np.array(w2d_kernel)

    
    w2darray = np.zeros(shape=(stepn, stepn))
    vel_array_xz = np.zeros(shape=(stepn, stepn))
    #ngas = len(ig[0])
    ngas = len(g_x)
    # Set the kernel weights, make sure to normalize properly
    htot = 0
    for i in range(ngas):
        htot += g_size[i] / ngas
    
       
    #print('First grid:', xcen - (stepnh*res), ycen + (stepnh*res),ycen - (stepnh*res))
    # Big loop
    for i in range(ngas):
        xi = (g_x[i] - xcen) / res_xz + stepnh
        zi = (g_z[i] - zcen) / res_xz + stepnh
        hi = 2 * g_size[i] / res_xz
        
        ixmin = int(np.floor(xi - hi)) + 1
        #print('x min',ixmin, xi, g_x[i])
        
        ixmin = max(ixmin, 0)
        ixmin = min(ixmin, stepn - 1)

        ixmax = int(np.floor(xi + hi))
        #print('x max',ixmax)
        ixmax = max(ixmax, 0)
        ixmax = min(ixmax, stepn - 1)

        izmin = int(np.floor(zi - hi)) + 1
        #print('y min',iymin, yi, g_y[i], 'y cen:',ycen, 'res and stepn',res,stepnh)
        izmin = max(izmin, 0)
        izmin = min(izmin, stepn - 1)

        izmax = int(np.floor(zi + hi))
        #print('y max',iymax)
        izmax = max(izmax, 0)
        izmax = min(izmax, stepn - 1)
        
        

        w2d_sum = 0.
        d2max = 4. * g_size[i] * g_size[i]
        kernfrac = nkernel * 0.5 / g_size[i]
        loopi = np.arange(ixmin, ixmax + 1, 1)
        loopj = np.arange(izmin, izmax + 1, 1)
        for ii in range(len(loopi)):
            r0xx = (ii - stepnh) * res_xz + xcen
            for jj in range(len(loopj)):
                r0zz = (jj - stepnh) * res_xz + zcen
                xx = g_x[i] - r0xx
                zz = g_z[i] - r0zz
                d2 = xx * xx + zz * zz
                if d2 <= d2max:
                    d = np.sqrt(d2)
                    xii = d * kernfrac
                    xxi = int(np.floor(xii))
                    w2d = (w2d_table[xxi + 1] - w2d_table[xxi]) * (xii - xxi) + w2d_table[xxi]
                    w2darray[ii, jj] = w2d
                    w2d_sum += w2d
                else:
                    w2darray[ii, jj] = 0.

        if w2d_sum > 0:
            weight = g_mass[i] / w2d_sum
            for ii in range(len(loopi)):
                for jj in range(len(loopj)):
                    if w2darray[ii, jj] > 0:
                        vel_array_xz[ii, jj] += w2darray[ii, jj] * weight
        else:
            ii = int(np.floor(xi + 0.5))
            jj = int(np.floor(zi + 0.5))
            if ((ii >=0) & (ii <= stepn-1) & (jj >= 0) & (jj <= stepn-1)):
                vel_array_xz[ii,jj] = vel_array_xz[ii,jj] + g_mass[i]
        
        i
    #return vel_array[0,:,:]
    extent_xz=(xcen - kpc_width, xcen + kpc_width, zcen - kpc_width, zcen + kpc_width)
    return vel_array_xz, res_xz, extent_xz

def compute_velocity_array_yz(xcen, ycen, zcen, coordinates, stepn = 16, pc_width = 40, nkernel = 200, kernel_file_path = '/home1/09528/amoac/'):
    g_y = coordinates['g_ys']
    g_z = coordinates['g_zs']
    g_mass = coordinates['g_mg']
    g_size = coordinates['g_hg']
    #print("g_y[0]", g_y[0])
    kpc_width = pc_width/1e3
    sizebox = 2*kpc_width
    #sizebox = pc_width
    res_yz = sizebox/stepn
    stepnh = (stepn-1)/2
    
    kernelfile = kernel_file_path + 'kernel2d'
    w2d_kernel = []
    w2d_table = np.zeros(nkernel+3)
    temp = open(kernelfile,'r').read().split('\n')
    

    for i in range(len(temp)-1):
        if (i%2 == 0):
            w2d_kernel.append(float(temp[i]))

    w2d_kernel.append(0.)
    w2d_kernel = np.array(w2d_kernel)

    
    w2darray = np.zeros(shape=(stepn, stepn))
    vel_array_yz = np.zeros(shape=(stepn, stepn))
    #ngas = len(ig[0])
    ngas = len(g_y)
    # Set the kernel weights, make sure to normalize properly
    htot = 0
    for i in range(ngas):
        htot += g_size[i] / ngas
    
       
    #print('First grid:', xcen - (stepnh*res), ycen + (stepnh*res),ycen - (stepnh*res))
    # Big loop
    for i in range(ngas):
        yi = (g_y[i] - ycen) / res_yz + stepnh
        zi = (g_z[i] - zcen) / res_yz + stepnh
        hi = 2 * g_size[i] / res_yz
        
        iymin = int(np.floor(yi - hi)) + 1
        #print('x min',ixmin, xi, g_x[i])
        
        iymin = max(iymin, 0)
        iymin = min(iymin, stepn - 1)

        iymax = int(np.floor(yi + hi))
        #print('x max',ixmax)
        iymax = max(iymax, 0)
        iymax = min(iymax, stepn - 1)

        izmin = int(np.floor(zi - hi)) + 1
        #print('y min',iymin, yi, g_y[i], 'y cen:',ycen, 'res and stepn',res,stepnh)
        izmin = max(izmin, 0)
        izmin = min(izmin, stepn - 1)

        izmax = int(np.floor(zi + hi))
        #print('y max',iymax)
        izmax = max(izmax, 0)
        izmax = min(izmax, stepn - 1)
        
        

        w2d_sum = 0.
        d2max = 4. * g_size[i] * g_size[i]
        kernfrac = nkernel * 0.5 / g_size[i]
        loopi = np.arange(iymin, iymax + 1, 1)
        loopj = np.arange(izmin, izmax + 1, 1)
        for ii in range(len(loopi)):
            r0yy = (ii - stepnh) * res_yz + ycen
            for jj in range(len(loopj)):
                r0zz = (jj - stepnh) * res_yz + zcen
                yy = g_y[i] - r0yy
                zz = g_z[i] - r0zz
                d2 = yy * yy + zz * zz
                if d2 <= d2max:
                    d = np.sqrt(d2)
                    yii = d * kernfrac
                    yyi = int(np.floor(yii))
                    w2d = (w2d_table[yyi + 1] - w2d_table[yyi]) * (yii - yyi) + w2d_table[yyi]
                    w2darray[ii, jj] = w2d
                    w2d_sum += w2d
                else:
                    w2darray[ii, jj] = 0.

        if w2d_sum > 0:
            weight = g_mass[i] / w2d_sum
            for ii in range(len(loopi)):
                for jj in range(len(loopj)):
                    if w2darray[ii, jj] > 0:
                        vel_array_yz[ii, jj] += w2darray[ii, jj] * weight
        else:
            ii = int(np.floor(yi + 0.5))
            jj = int(np.floor(zi + 0.5))
            if ((ii >=0) & (ii <= stepn-1) & (jj >= 0) & (jj <= stepn-1)):
                vel_array_yz[ii,jj] = vel_array_yz[ii,jj] + g_mass[i]
        
        i
    #return vel_array[0,:,:]
    extent_yz=(ycen - kpc_width, ycen + kpc_width, zcen - kpc_width, zcen + kpc_width)
    return vel_array_yz, res_yz, extent_yz