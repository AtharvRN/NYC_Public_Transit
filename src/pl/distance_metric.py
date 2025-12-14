import numpy as np
def haversine_distance(lat1, lng1, lat2, lng2):
    """
    Great-circle distance between two points (lat1, lng1) and (lat2, lng2).
    Inputs in degrees. Output in meters.
    """
    # print(lat1, lng1)
    # print(lat2, lng2)
    if lat1 == None or lng1 == None or lat2 == None or lng2 == None:
        return None
    R = 6371000  # Earth radius in meters
    # print(lat1, lng1)
    # print(lat2, lng2)
    # # convert to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    lam1, lam2 = np.radians(lng1), np.radians(lng2)

    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    haversine_dist = R * c
    return haversine_dist