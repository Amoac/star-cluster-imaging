import numpy as np

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
    


