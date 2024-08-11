import numpy as np

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
    
    return g_ind[0], s_ind[0] 


