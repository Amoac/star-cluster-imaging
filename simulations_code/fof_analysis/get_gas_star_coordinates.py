import numpy as np

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
