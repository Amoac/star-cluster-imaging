import numpy as np

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
