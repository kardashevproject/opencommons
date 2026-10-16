"""Trans-Oceanic Geodesic Trajectory Planning Benchmark.

Composes 6 verified OpenCommons concepts from the geodesy_and_spatial domain:
1. `andoyer_lambert_geodesic_distance`: Ellipsoidal reference distance on WGS-84 reference ellipsoid.
2. `great_circle_bearing`: Initial navigation course heading between departure and arrival coordinates.
3. `intermediate_point_geodesic`: Exact spherical trigonometry fractional arc interpolation.
4. `point_in_polygon_winding_number`: Non-zero winding number test for restricted airspace / storm geofencing.
5. `douglas_peucker_simplify_step`: Recursive polyline decimation preserving corridor envelope within epsilon.
6. `morton_encode_2d`: Z-order space-filling curve 2D spatial indexing for spatial database storage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "geodesy_and_spatial.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
andoyer_lambert_geodesic_distance = _FUNCS["andoyer_lambert_geodesic_distance"]
great_circle_bearing = _FUNCS["great_circle_bearing"]
intermediate_point_geodesic = _FUNCS["intermediate_point_geodesic"]
point_in_polygon_winding_number = _FUNCS["point_in_polygon_winding_number"]
douglas_peucker_simplify_step = _FUNCS["douglas_peucker_simplify_step"]
morton_encode_2d = _FUNCS["morton_encode_2d"]

# WGS-84 Reference Ellipsoid Constants
WGS84_SEMI_MAJOR = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563


def plan_transoceanic_geodesic_trajectory(
    origin: tuple[float, float],
    destination: tuple[float, float],
    num_waypoints: int = 25,
    restricted_geofences: list[tuple[list[float], list[float]]] | None = None,
    simplify_epsilon_deg: float = 0.5,
) -> dict[str, Any]:
    """Plans an optimal trans-oceanic flight trajectory with geofence collision detection and compression.

    Args:
        origin: (latitude_deg, longitude_deg) of departure airport
        destination: (latitude_deg, longitude_deg) of arrival airport
        num_waypoints: Number of intermediate waypoints along geodesic arc
        restricted_geofences: List of (poly_lats, poly_lons) defining restricted airspace zones
        simplify_epsilon_deg: Maximum allowed deviation for Douglas-Peucker polyline simplification

    Returns:
        Dictionary containing route distance, initial bearing, waypoints, geofence violations,
        simplified waypoint indices, and Morton spatial keys.
    """
    lat1, lon1 = origin
    lat2, lon2 = destination

    # 1. Total reference ellipsoidal distance on WGS-84
    total_distance_m = andoyer_lambert_geodesic_distance(
        lat1, lon1, lat2, lon2, WGS84_SEMI_MAJOR, WGS84_FLATTENING
    )

    # 2. Initial course bearing
    initial_bearing_deg = great_circle_bearing(lat1, lon1, lat2, lon2)

    # 3. Intermediate waypoints along great-circle trajectory
    waypoints: list[tuple[float, float]] = []
    for i in range(num_waypoints):
        frac = i / max(1, num_waypoints - 1)
        w_lat, w_lon = intermediate_point_geodesic(lat1, lon1, lat2, lon2, frac)
        waypoints.append((w_lat, w_lon))

    # 4. Airspace collision check against restricted geofences
    geofence_violations: list[dict[str, Any]] = []
    if restricted_geofences:
        for zone_idx, (poly_lats, poly_lons) in enumerate(restricted_geofences):
            for wp_idx, (w_lat, w_lon) in enumerate(waypoints):
                is_inside = point_in_polygon_winding_number(w_lat, w_lon, poly_lats, poly_lons)
                if is_inside:
                    geofence_violations.append({
                        "zone_index": zone_idx,
                        "waypoint_index": wp_idx,
                        "coordinate": (w_lat, w_lon),
                    })

    # 5. Waypoint polyline simplification via Douglas-Peucker
    w_lats = [wp[0] for wp in waypoints]
    w_lons = [wp[1] for wp in waypoints]
    kept_indices = douglas_peucker_simplify_step(w_lats, w_lons, simplify_epsilon_deg)
    simplified_waypoints = [waypoints[idx] for idx in kept_indices]

    # 6. Spatial indexing via Morton Z-order curve
    # Normalize coordinates to 16-bit integer grid [0, 65535]
    morton_keys: list[int] = []
    for w_lat, w_lon in simplified_waypoints:
        x_norm = int(max(0.0, min(1.0, (w_lon + 180.0) / 360.0)) * 65535)
        y_norm = int(max(0.0, min(1.0, (w_lat + 90.0) / 180.0)) * 65535)
        m_key = morton_encode_2d(x_norm, y_norm)
        morton_keys.append(m_key)

    return {
        "origin": origin,
        "destination": destination,
        "total_distance_m": total_distance_m,
        "total_distance_km": total_distance_m / 1000.0,
        "initial_bearing_deg": initial_bearing_deg,
        "num_raw_waypoints": len(waypoints),
        "waypoints": waypoints,
        "geofence_violations": geofence_violations,
        "simplified_indices": kept_indices,
        "simplified_waypoints": simplified_waypoints,
        "compression_ratio": len(simplified_waypoints) / len(waypoints),
        "morton_keys": morton_keys,
    }
