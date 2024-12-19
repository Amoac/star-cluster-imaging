import numpy as np

def get_star_properties(part, snapshot):
    """
    Extracts star properties from the input data and organizes them into a dictionary, keyed by snapshot.

    Parameters:
    part (dict): Dictionary containing star particle data.
    snapshot (str): Snapshot identifier.

    Returns:
    dict: A dictionary containing snapshot-specific star properties.

    Example:
    >>> star_props = get_star_properties(part, 'snapshot_001')
    >>> print(star_props['x_snapshot_001'])
    """
    print(part)
    # Extract star properties from the input data
  
    # Create a dictionary with the snapshot-specific star properties
 return {
        'rxyz': part['star'].prop('host.distance.total'),  # Total distance of star particles
        'x': part['star'].prop('host.distance.principal.cartesian')[:, 0]  # x-coordinates of star particles
        'y': part['star'].prop('host.distance.principal.cartesian')[:, 1]  # y-coordinates of star particles
        'z': part['star'].prop('host.distance.principal.cartesian')[:, 2]  # z-coordinates of star particles
        'Rxy': part['star'].prop('host.distance.principal.cylindrical')[:, 0]  # Cylindrical radial distance of star particles
        'vR': part['star'].prop('host.velocity.principal.cylindrical')[:, 0]  # Radial velocity of star particles
        'vphi': part['star'].prop('host.velocity.principal.cylindrical')[:, 2]  # Azimuthal velocity of star particles
        'vz2': part['star'].prop('host.velocity.principal.cylindrical')[:, 1]  # Vertical velocity of star particles
        'vx': part['star'].prop('host.velocity.principal.cartesian')[:, 0]  # x-component of velocity of star particles
        'vy': part['star'].prop('host.velocity.principal.cartesian')[:, 1]  # y-component of velocity of star particles
        'vz': part['star'].prop('host.velocity.principal.cartesian')[:, 2]  # z-component of velocity of star particles
        'age': part['star'].prop('age')  # Age of star particles
        'mass':part['star'].prop('mass')  # Mass of star particles
        'feh': part['star'].prop('metallicity.fe')  # Iron metallicity of star particles
        'ids':  part['star']['id']  # IDs of star particles
        'id_generation': part['star']['id.generation']  # Generation ID of star particles
        'id_child': part['star']['id.child']  # Child ID of star particles
    }
    
    # Return the dictionary
    return star_properties
