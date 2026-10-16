"""Pytest test suite for Geodesy and Spatial Geometry Trajectory Planning Benchmark."""

import json
from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.geodesy_and_spatial.benchmark import (
    plan_transoceanic_geodesic_trajectory,
    andoyer_lambert_geodesic_distance,
    great_circle_bearing,
    intermediate_point_geodesic,
    point_in_polygon_winding_number,
    douglas_peucker_simplify_step,
    morton_encode_2d,
    WGS84_SEMI_MAJOR,
    WGS84_FLATTENING,
)


def test_transoceanic_geodesic_distance_and_bearing():
    """Verify geodesic distance and initial bearing calculation between JFK and LHR."""
    jfk = (40.6413, -73.7781)
    lhr = (51.4700, -0.4543)

    plan = plan_transoceanic_geodesic_trajectory(
        origin=jfk,
        destination=lhr,
        num_waypoints=30,
    )

    # Standard WGS-84 distance between JFK and LHR is ~5570 km
    assert 5500.0 < plan["total_distance_km"] < 5650.0, f"Distance out of range: {plan['total_distance_km']}"

    # Initial great circle departure bearing from JFK towards LHR is ~51.4 degrees
    assert 48.0 < plan["initial_bearing_deg"] < 55.0, f"Bearing out of range: {plan['initial_bearing_deg']}"


def test_intermediate_waypoints_endpoints_and_continuity():
    """Verify intermediate waypoints match origin and destination at endpoints."""
    origin = (37.7749, -122.4194)  # San Francisco
    destination = (35.6762, 139.6503)  # Tokyo

    plan = plan_transoceanic_geodesic_trajectory(
        origin=origin,
        destination=destination,
        num_waypoints=20,
    )

    waypoints = plan["waypoints"]
    assert len(waypoints) == 20

    # Start waypoint matches origin
    assert abs(waypoints[0][0] - origin[0]) < 1e-4
    assert abs(waypoints[0][1] - origin[1]) < 1e-4

    # End waypoint matches destination
    assert abs(waypoints[-1][0] - destination[0]) < 1e-4
    assert abs(waypoints[-1][1] - destination[1]) < 1e-4

    # Trajectory reaches higher northern latitudes (Great Circle path over North Pacific / Aleutians)
    max_lat = max(wp[0] for wp in waypoints)
    assert max_lat > 45.0, f"Trans-Pacific great circle should curve north into sub-arctic, got max lat: {max_lat}"
    assert max_lat > max(origin[0], destination[0]) + 8.0, "Great circle route should peak substantially north of endpoints"



def test_geofence_airspace_collision_detection():
    """Verify non-zero winding number polygon collision detection on restricted airspace."""
    jfk = (40.6413, -73.7781)
    lhr = (51.4700, -0.4543)

    # Define a restricted airspace zone (bounding box over mid-Atlantic near latitude 50N, longitude -30W)
    # Polygon vertices (latitudes, longitudes) in counter-clockwise order
    restricted_zone = (
        [48.0, 48.0, 54.0, 54.0],
        [-35.0, -25.0, -25.0, -35.0],
    )

    plan = plan_transoceanic_geodesic_trajectory(
        origin=jfk,
        destination=lhr,
        num_waypoints=50,
        restricted_geofences=[restricted_zone],
    )

    # Trans-Atlantic flight from JFK to LHR passes directly through 50N, -30W
    violations = plan["geofence_violations"]
    assert len(violations) > 0, "Flight path should intersect mid-Atlantic restricted zone"
    for v in violations:
        assert v["zone_index"] == 0
        lat, lon = v["coordinate"]
        assert 48.0 <= lat <= 54.0
        assert -35.0 <= lon <= -25.0


def test_douglas_peucker_waypoint_compression():
    """Verify Douglas-Peucker algorithm compresses dense waypoints within tolerance."""
    origin = (0.0, 0.0)
    destination = (0.0, 50.0)

    # Route along the equator with 100 dense waypoints
    plan = plan_transoceanic_geodesic_trajectory(
        origin=origin,
        destination=destination,
        num_waypoints=100,
        simplify_epsilon_deg=0.1,
    )

    # Perfectly linear route along equator compresses from 100 to only 2 endpoints!
    assert len(plan["simplified_waypoints"]) == 2
    assert plan["compression_ratio"] == 0.02
    assert plan["simplified_indices"] == [0, 99]


def test_morton_encoding_spatial_properties():
    """Verify Morton Z-order curve bit interleaving."""
    # Origin (0, 0) -> Morton code 0
    assert morton_encode_2d(0, 0) == 0

    # (1, 0) -> x bit 0 -> code 1
    # (0, 1) -> y bit 0 -> code 2
    # (1, 1) -> x bit 0, y bit 0 -> code 3
    assert morton_encode_2d(1, 0) == 1
    assert morton_encode_2d(0, 1) == 2
    assert morton_encode_2d(1, 1) == 3

    # (2, 0) -> x bit 1 -> code 4
    assert morton_encode_2d(2, 0) == 4


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in geodesy_and_spatial.json."""
    cat_path = repo_root / "data" / "domains" / "geodesy_and_spatial.json"
    assert cat_path.is_file()
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "andoyer_lambert_geodesic_distance",
        "great_circle_bearing",
        "intermediate_point_geodesic",
        "point_in_polygon_winding_number",
        "douglas_peucker_simplify_step",
        "morton_encode_2d",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
