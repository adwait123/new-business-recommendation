import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import time
import math
import tempfile
import zipfile
from io import BytesIO
import base64

# Only import these if we're running in Streamlit context
try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Point = None
    Polygon = None

try:
    from geopy.distance import geodesic

    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    geodesic = None

try:
    import folium
    from branca.element import Element

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    folium = None


# Check if we're running in Streamlit context
def is_streamlit_context():
    """Check if we're running in Streamlit context"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except:
        return False


# Set page configuration only if in Streamlit context
if is_streamlit_context():
    st.set_page_config(
        page_title="ZIP Code Polygon Generator",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS for better styling
if is_streamlit_context():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            margin: 1rem 0;
        }
        .error-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            margin: 1rem 0;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


def log_debug(message):
    """Helper function for debug logging"""
    if is_streamlit_context():
        if st.session_state.get('debug_mode', False):
            st.write(f"🐛 DEBUG: {message}")
    else:
        print(f"DEBUG: {message}")


def graham_scan(points):
    """Graham scan algorithm to find convex hull of points."""
    if len(points) < 3:
        return points

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def polar_angle(p0, p1):
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def distance_squared(p0, p1):
        return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2

    start = min(points, key=lambda p: (p[1], p[0]))

    def compare_points(p):
        angle = polar_angle(start, p)
        dist = distance_squared(start, p)
        return (angle, dist)

    sorted_points = sorted([p for p in points if p != start], key=compare_points)

    if len(sorted_points) == 0:
        return [start]

    hull = [start, sorted_points[0]]

    for p in sorted_points[1:]:
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def create_folium_visualization(results_df, zip_db_df, assignments_df, covered_df, uncovered_df):
    """Create Folium visualization and return HTML content"""
    if not FOLIUM_AVAILABLE:
        raise ImportError("Folium is required for visualization. Install with: pip install folium")

    # Create ZIP coordinate lookup
    zip_coordinates = {}
    for _, row in zip_db_df.iterrows():
        zip_coordinates[int(row['zip'])] = (row['lat'], row['lng'])

    # Calculate center of map
    if len(zip_coordinates) > 0:
        all_coords = list(zip_coordinates.values())
        mean_lat = sum([c[0] for c in all_coords]) / len(all_coords)
        mean_lng = sum([c[1] for c in all_coords]) / len(all_coords)
    else:
        mean_lat, mean_lng = 39.8283, -98.5795  # Center of US

    # Create map
    m = folium.Map(location=[mean_lat, mean_lng], zoom_start=5, tiles='cartodbpositron')

    # Calculate summary statistics
    total_covered = len(covered_df)
    total_uncovered = len(uncovered_df)
    total_zips = total_covered + total_uncovered
    coverage_pct = (total_covered / total_zips * 100) if total_zips > 0 else 0

    # Add summary to map
    summary_html = f'''
    <div style="position: fixed; z-index: 9999; background: white; padding: 10px; border: 2px solid #333; top: 10px; left: 50%; transform: translateX(-50%); font-size: 16px;">
        <b>Total Coverage:</b> {coverage_pct:.1f}%<br>
        <b>ZIPs covered:</b> {total_covered:,} / {total_zips:,}<br>
        <b>Algorithm:</b> Graham Scan Convex Hull (Max 6 Points)<br>
        <div style="margin-top: 8px; font-size: 14px;">
            <span style="color: green; background: lightgreen; padding: 2px;">●</span> Covered ZIPs &nbsp;
            <span style="color: red; background: pink; padding: 2px;">●</span> Uncovered ZIPs &nbsp;
            <span style="color: blue;">▲</span> Business Centers
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(summary_html))

    # Add business polygons
    for _, row in results_df.iterrows():
        if row['has_valid_polygon'] == 'Yes' and row['boundary_zips']:
            business_id = row['business_id']
            biz_zip = int(row['biz_zip'])

            # Parse boundary ZIPs
            boundary_zips = []
            if ';' in str(row['boundary_zips']):
                boundary_zips = [int(z.strip()) for z in str(row['boundary_zips']).split(';') if z.strip().isdigit()]
            else:
                boundary_zips = [int(z.strip()) for z in str(row['boundary_zips']).split(',') if z.strip().isdigit()]

            # Create polygon points
            poly_points = []
            for zip_code in boundary_zips:
                if zip_code in zip_coordinates:
                    poly_points.append(zip_coordinates[zip_code])

            # Draw polygon if we have enough points
            if len(poly_points) >= 3:
                popup_html = f"""
                <b>Business ID:</b> {business_id}<br>
                <b>Business ZIP:</b> {biz_zip}<br>
                <b>Boundary ZIPs:</b> {'; '.join(map(str, boundary_zips))}<br>
                <b>Coverage:</b> {row['coverage_percentage']}%<br>
                <b>Contained ZIPs:</b> {row['contained_zip_count']}<br>
                <b>Total Assigned:</b> {row['total_assigned_zips']}
                """

                folium.Polygon(
                    locations=poly_points,
                    color='blue',
                    fill=True,
                    fill_opacity=0.2,
                    weight=2,
                    tooltip=f"Business {business_id}",
                    popup=folium.Popup(popup_html, max_width=350)
                ).add_to(m)

            # Add business center marker
            if biz_zip in zip_coordinates:
                folium.Marker(
                    location=zip_coordinates[biz_zip],
                    icon=folium.Icon(color='blue', icon='building'),
                    popup=folium.Popup(f"Business {business_id}<br>ZIP: {biz_zip}", max_width=200),
                    tooltip=f"Business {business_id}"
                ).add_to(m)

    # Add covered ZIP markers
    covered_zips_added = set()  # Avoid duplicates
    for _, row in covered_df.iterrows():
        zip_code = int(row['target_zip'])
        if zip_code in zip_coordinates and zip_code not in covered_zips_added:
            folium.CircleMarker(
                location=zip_coordinates[zip_code],
                radius=3,
                color='darkgreen',
                fill=True,
                fill_color='lightgreen',
                fill_opacity=0.8,
                weight=1,
                tooltip=f"Covered ZIP: {zip_code}<br>Business: {row['business_id']}"
            ).add_to(m)
            covered_zips_added.add(zip_code)

    # Add uncovered ZIP markers
    uncovered_zips_added = set()  # Avoid duplicates
    for _, row in uncovered_df.iterrows():
        zip_code = int(row['target_zip'])
        if zip_code in zip_coordinates and zip_code not in uncovered_zips_added:
            folium.CircleMarker(
                location=zip_coordinates[zip_code],
                radius=3,
                color='darkred',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                weight=1,
                tooltip=f"Uncovered ZIP: {zip_code}<br>Business: {row['business_id']}<br>Reason: {row.get('reason', 'Unknown')}"
            ).add_to(m)
            uncovered_zips_added.add(zip_code)

    # Return HTML content
    return m._repr_html_()


def simple_distance_fallback(coord1, coord2):
    """Simple Euclidean distance calculation as fallback when geopy is not available"""
    # Convert to approximate miles using lat/lng differences
    lat_diff = coord1[0] - coord2[0]
    lng_diff = coord1[1] - coord2[1]
    # Rough approximation: 1 degree ≈ 69 miles
    return math.sqrt(lat_diff ** 2 + lng_diff ** 2) * 69


def process_polygon_generation(assignments_df, zip_db_df, progress_bar=None, status_text=None):
    """Main function to process polygon generation with optional Streamlit integration"""

    def update_progress(progress, message):
        """Update progress indicators if available"""
        if progress_bar is not None:
            progress_bar.progress(progress)
        if status_text is not None:
            status_text.text(message)
        elif not is_streamlit_context():
            print(message)

    # Create optimized lookup
    zip_coordinates = {}
    for _, row in zip_db_df.iterrows():
        zip_int = int(row['zip'])
        zip_coordinates[zip_int] = (row['lat'], row['lng'])

    update_progress(0.1, f"Created coordinate lookup for {len(zip_coordinates)} ZIP codes")

    # Group assignments by business
    business_groups = {}
    for _, row in assignments_df.iterrows():
        business_id = row['business_id']
        biz_zip = int(row['biz_zip'])
        group_key = business_id

        if group_key not in business_groups:
            business_groups[group_key] = {
                'business_id': business_id,
                'biz_zip': biz_zip,
                'assignments': []
            }

        business_groups[group_key]['assignments'].append({
            'zip': int(row['target_zip']),
            'distance': row['distance_miles']
        })

    update_progress(0.2, f"Found {len(business_groups)} business groups")

    # Distance calculation with caching
    distance_cache = {}

    def cached_distance(zip1, zip2):
        cache_key = (min(zip1, zip2), max(zip1, zip2))
        if cache_key in distance_cache:
            return distance_cache[cache_key]

        if zip1 not in zip_coordinates or zip2 not in zip_coordinates:
            return float('inf')

        coords1 = zip_coordinates[zip1]
        coords2 = zip_coordinates[zip2]

        # Use geopy if available, otherwise fallback to simple calculation
        if GEOPY_AVAILABLE:
            distance = geodesic(coords1, coords2).miles
        else:
            # Fallback to simple Euclidean distance approximation
            distance = simple_distance_fallback(coords1, coords2)
            log_debug(f"Using fallback distance calculation for {zip1} to {zip2}: {distance:.2f} miles")

        distance_cache[cache_key] = distance
        return distance

    def validate_distance_constraint(zips, max_distance=85):
        for i in range(len(zips)):
            if zips[i] not in zip_coordinates:
                continue
            for j in range(i + 1, len(zips)):
                if zips[j] not in zip_coordinates:
                    continue
                distance = cached_distance(zips[i], zips[j])
                if distance > max_distance:
                    return False, {'zip1': zips[i], 'zip2': zips[j], 'distance': distance}
        return True, None

    def count_contained_zips(polygon_zips, all_zips):
        if len(polygon_zips) < 3:
            return 0, []

        if not SHAPELY_AVAILABLE:
            log_debug("Shapely not available, using bounding box approximation for polygon containment")
            # Fallback: use bounding box approximation
            polygon_coords = [zip_coordinates[z] for z in polygon_zips if z in zip_coordinates]
            if len(polygon_coords) < 3:
                return 0, []

            # Calculate bounding box
            min_lat = min(coord[0] for coord in polygon_coords)
            max_lat = max(coord[0] for coord in polygon_coords)
            min_lng = min(coord[1] for coord in polygon_coords)
            max_lng = max(coord[1] for coord in polygon_coords)

            # Check which ZIPs fall within bounding box
            contained_zips = []
            for zip_code in all_zips:
                if zip_code in zip_coordinates:
                    lat, lng = zip_coordinates[zip_code]
                    if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
                        contained_zips.append(zip_code)

            return len(contained_zips), contained_zips

        # Use Shapely for accurate polygon containment
        polygon_points = []
        for zip_code in polygon_zips:
            if zip_code in zip_coordinates:
                lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
                polygon_points.append((lng, lat))

        if len(polygon_points) < 3:
            return 0, []

        try:
            polygon = Polygon(polygon_points)
            if not polygon.is_valid:
                polygon_points.reverse()
                polygon = Polygon(polygon_points)
        except Exception:
            return 0, []

        contained_zips = []
        minx, miny, maxx, maxy = polygon.bounds

        for zip_code in all_zips:
            if zip_code not in zip_coordinates:
                continue

            lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
            if minx <= lng <= maxx and miny <= lat <= maxy:
                point = Point(lng, lat)
                if polygon.contains(point) or polygon.touches(point):
                    contained_zips.append(zip_code)

        return len(contained_zips), contained_zips

    def create_constrained_convex_hull(business_zip, valid_zips, max_distance=85, max_points=6):
        min_points = 3
        if len(valid_zips) < min_points:
            return [], []

        # Sample for performance if too many ZIPs
        sample_size = 500
        working_zips = valid_zips
        if len(valid_zips) > sample_size:
            import random
            random.seed(42)
            if business_zip in valid_zips and business_zip in zip_coordinates:
                valid_zips_without_business = [z for z in valid_zips if z != business_zip]
                if len(valid_zips_without_business) > sample_size - 1:
                    sampled_zips = random.sample(valid_zips_without_business, sample_size - 1)
                    sampled_zips.append(business_zip)
                else:
                    sampled_zips = valid_zips
            else:
                sampled_zips = random.sample(valid_zips, min(sample_size, len(valid_zips)))
            working_zips = sampled_zips

        valid_zip_coords = [z for z in working_zips if z in zip_coordinates]

        # Build distance-constrained clusters
        clusters = []
        for center_zip in valid_zip_coords:
            cluster = [center_zip]
            for other_zip in valid_zip_coords:
                if other_zip == center_zip:
                    continue

                valid_for_cluster = True
                for cluster_zip in cluster:
                    if cached_distance(other_zip, cluster_zip) > max_distance:
                        valid_for_cluster = False
                        break

                if valid_for_cluster:
                    cluster.append(other_zip)

            if len(cluster) >= min_points:
                clusters.append(cluster)

        if not clusters:
            return [], []

        best_hull_zips = []
        best_contained_zips = []
        best_coverage = 0

        for cluster in clusters[:10]:  # Limit for performance
            cluster_points = []
            zip_to_point = {}
            for zip_code in cluster:
                if zip_code in zip_coordinates:
                    lat, lng = zip_coordinates[zip_code]
                    point = (lng, lat)
                    cluster_points.append(point)
                    zip_to_point[point] = zip_code

            if len(cluster_points) < min_points:
                continue

            hull_points = graham_scan(cluster_points)

            if len(hull_points) > max_points:
                from itertools import combinations
                best_subset_coverage = 0
                best_subset_hull = []
                best_subset_contained = []

                if len(hull_points) > 12:
                    step = len(hull_points) // max_points
                    hull_subset = [hull_points[i] for i in range(0, len(hull_points), step)][:max_points]
                    hull_combinations = [hull_subset]
                else:
                    hull_combinations = list(combinations(hull_points, max_points))[:50]

                for hull_subset in hull_combinations:
                    subset_hull_zips = [zip_to_point[point] for point in hull_subset]
                    is_valid, violation = validate_distance_constraint(subset_hull_zips, max_distance)
                    if not is_valid:
                        continue

                    coverage_count, contained = count_contained_zips(subset_hull_zips, valid_zips)
                    if coverage_count > best_subset_coverage:
                        best_subset_coverage = coverage_count
                        best_subset_hull = subset_hull_zips
                        best_subset_contained = contained

                hull_zips = best_subset_hull
                coverage_count = best_subset_coverage
                contained = best_subset_contained

                if not hull_zips:
                    continue
            else:
                hull_zips = [zip_to_point[point] for point in hull_points]
                is_valid, violation = validate_distance_constraint(hull_zips, max_distance)
                if not is_valid:
                    continue
                coverage_count, contained = count_contained_zips(hull_zips, valid_zips)

            if coverage_count > best_coverage:
                best_coverage = coverage_count
                best_hull_zips = hull_zips
                best_contained_zips = contained

        return best_hull_zips, best_contained_zips

    # Process each business
    results = []
    total_businesses = len(business_groups)

    for i, (business_id, group) in enumerate(business_groups.items()):
        progress = 0.2 + (0.7 * (i + 1) / total_businesses)
        update_progress(progress, f"Processing business {i + 1}/{total_businesses}: {business_id}")

        business_zip = group['biz_zip']
        all_assigned_zips = [a['zip'] for a in group['assignments']]

        # Filter valid ZIPs
        valid_zips = []
        business_coords = zip_coordinates.get(business_zip)

        for assignment in group['assignments']:
            zip_code = assignment['zip']
            if zip_code in zip_coordinates:
                if business_coords:
                    # Calculate distance using available method
                    if GEOPY_AVAILABLE:
                        distance = geodesic(business_coords, zip_coordinates[zip_code]).miles
                    else:
                        distance = simple_distance_fallback(business_coords, zip_coordinates[zip_code])

                    if distance <= 85:
                        valid_zips.append(zip_code)
                else:
                    if assignment['distance'] <= 85:
                        valid_zips.append(zip_code)

        min_points = 3
        if len(valid_zips) < min_points:
            results.append({
                'business_id': group['business_id'],
                'biz_zip': group['biz_zip'],
                'boundary_zips': "",
                'contained_zip_count': 0,
                'contained_zips': [],
                'total_assigned_zips': len(all_assigned_zips),
                'within_distance_constraint': len(valid_zips),
                'coverage_percentage': 0,
                'has_valid_polygon': "No",
                'polygon_point_count': 0,
                'reason': f"Insufficient ZIPs within distance constraint (need at least {min_points})"
            })
            continue

        optimal_polygon, contained_zips = create_constrained_convex_hull(business_zip, valid_zips, max_distance=85,
                                                                         max_points=6)

        if len(optimal_polygon) < min_points:
            results.append({
                'business_id': group['business_id'],
                'biz_zip': group['biz_zip'],
                'boundary_zips': "",
                'contained_zip_count': 0,
                'contained_zips': [],
                'total_assigned_zips': len(all_assigned_zips),
                'within_distance_constraint': len(valid_zips),
                'coverage_percentage': 0,
                'has_valid_polygon': "No",
                'polygon_point_count': 0,
                'reason': f"Could not create valid convex hull with at least {min_points} points"
            })
            continue

        is_valid, violation = validate_distance_constraint(optimal_polygon, max_distance=85)

        if not is_valid:
            results.append({
                'business_id': group['business_id'],
                'biz_zip': group['biz_zip'],
                'boundary_zips': "",
                'contained_zip_count': 0,
                'contained_zips': [],
                'total_assigned_zips': len(all_assigned_zips),
                'within_distance_constraint': len(valid_zips),
                'coverage_percentage': 0,
                'max_polygon_distance': round(violation['distance'], 2),
                'has_valid_polygon': "No",
                'polygon_point_count': 0,
                'reason': "Invalid convex hull - distance constraint violation"
            })
            continue

        coverage_percentage = round((len(contained_zips) / len(valid_zips)) * 85) if valid_zips else 0

        # Calculate max distance between polygon points
        max_polygon_distance = 0
        for i in range(len(optimal_polygon)):
            if optimal_polygon[i] not in zip_coordinates:
                continue
            for j in range(i + 1, len(optimal_polygon)):
                if optimal_polygon[j] not in zip_coordinates:
                    continue
                distance = cached_distance(optimal_polygon[i], optimal_polygon[j])
                if distance > max_polygon_distance:
                    max_polygon_distance = distance

        results.append({
            'business_id': group['business_id'],
            'biz_zip': group['biz_zip'],
            'boundary_zips': ';'.join(map(str, optimal_polygon)),
            'contained_zip_count': len(contained_zips),
            'contained_zips': contained_zips,
            'total_assigned_zips': len(all_assigned_zips),
            'within_distance_constraint': len(valid_zips),
            'coverage_percentage': coverage_percentage,
            'max_polygon_distance': round(max_polygon_distance, 2) if max_polygon_distance > 0 else 0,
            'has_valid_polygon': "Yes",
            'polygon_point_count': len(optimal_polygon)
        })

    update_progress(1.0, "Processing complete!")

    return results


def main():
    """Main Streamlit application"""
    if not is_streamlit_context():
        print("❌ This script must be run with: streamlit run streamlit_polygon_generation.py")
        return

    # Check for required libraries and show warnings
    missing_libraries = []
    if not SHAPELY_AVAILABLE:
        missing_libraries.append("shapely")
    if not GEOPY_AVAILABLE:
        missing_libraries.append("geopy")

    if missing_libraries:
        st.warning(f"⚠️ Some optional libraries are missing: {', '.join(missing_libraries)}")
        st.info("💡 For full functionality, install with: pip install " + " ".join(missing_libraries))

        if not SHAPELY_AVAILABLE:
            st.warning("🔺 Without Shapely: Using bounding box approximation for polygon containment (less accurate)")
        if not GEOPY_AVAILABLE:
            st.warning("📐 Without GeoPy: Using simple distance calculation (less accurate)")

        # Show continue option
        if not st.checkbox("Continue with limited functionality", value=False):
            st.stop()

    if not FOLIUM_AVAILABLE:
        st.info("🗺️ Folium not available - map visualization will be disabled. Install with: pip install folium")

    st.markdown('<h1 class="main-header">🗺️ ZIP Code Polygon Generator</h1>', unsafe_allow_html=True)

    st.markdown("""
    This application generates optimal ZIP code polygons using the Graham Scan convex hull algorithm 
    with distance constraints. Upload your data files to get started.
    """)

    # Show library status
    with st.expander("📚 Library Status"):
        col1, col2, col3 = st.columns(3)

        with col1:
            status = "✅ Available" if SHAPELY_AVAILABLE else "❌ Missing"
            st.write(f"**Shapely:** {status}")
            if not SHAPELY_AVAILABLE:
                st.caption("Used for accurate polygon calculations")

        with col2:
            status = "✅ Available" if GEOPY_AVAILABLE else "❌ Missing"
            st.write(f"**GeoPy:** {status}")
            if not GEOPY_AVAILABLE:
                st.caption("Used for precise distance calculations")

        with col3:
            status = "✅ Available" if FOLIUM_AVAILABLE else "❌ Missing"
            st.write(f"**Folium:** {status}")
            if not FOLIUM_AVAILABLE:
                st.caption("Used for interactive map visualization")

    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("📁 File Uploads")

        # File upload widgets
        assignments_file = st.file_uploader(
            "ZIP Code Assignments CSV",
            type=['csv'],
            help="CSV file containing business_id, biz_zip, target_zip, distance_miles columns"
        )

        zip_db_file = st.file_uploader(
            "ZIP Code Database CSV",
            type=['csv'],
            help="CSV file containing zip, lat, lng columns"
        )

        st.header("⚙️ Settings")

        max_distance = st.slider(
            "Maximum Distance (miles)",
            min_value=50,
            max_value=150,
            value=85,
            help="Maximum distance constraint between polygon points"
        )

        max_points = st.slider(
            "Maximum Polygon Points",
            min_value=3,
            max_value=10,
            value=6,
            help="Maximum number of points in each polygon"
        )

        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Enable debug output"
        )

        if debug_mode:
            st.session_state['debug_mode'] = True

    # Main content area
    if assignments_file is not None and zip_db_file is not None:
        try:
            # Load data with improved CSV handling
            with st.spinner("Loading data files..."):
                # Read assignments file with flexible handling
                assignments_df = pd.read_csv(assignments_file)
                # Clean column names (remove whitespace, standardize)
                assignments_df.columns = assignments_df.columns.str.strip().str.lower().str.replace(' ', '_')

                # Read ZIP database file
                zip_db_df = pd.read_csv(zip_db_file)
                # Clean column names
                zip_db_df.columns = zip_db_df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Display actual column names for debugging
            with st.expander("🔍 Debug: Actual Column Names"):
                st.write("**Assignments file columns:**", list(assignments_df.columns))
                st.write("**ZIP database columns:**", list(zip_db_df.columns))

            # Validate data structure with flexible column matching
            required_assignments_cols = ['business_id', 'biz_zip', 'target_zip', 'distance_miles']
            required_zip_db_cols = ['zip', 'lat', 'lng']

            # Check for exact matches first, then try common variations
            def find_column_match(df, target_col):
                """Find matching column with flexible naming"""
                columns = df.columns.tolist()

                # Direct match
                if target_col in columns:
                    return target_col

                # Common variations
                variations = {
                    'business_id': ['businessid', 'business_id', 'biz_id', 'id'],
                    'biz_zip': ['biz_zip', 'business_zip', 'bizzip', 'zip'],
                    'target_zip': ['target_zip', 'targetzip', 'target', 'destination_zip'],
                    'distance_miles': ['distance_miles', 'distance', 'miles', 'dist'],
                    'zip': ['zip', 'zipcode', 'zip_code', 'postal_code'],
                    'lat': ['lat', 'latitude', 'y'],
                    'lng': ['lng', 'lon', 'longitude', 'x']
                }

                if target_col in variations:
                    for variation in variations[target_col]:
                        if variation in columns:
                            return variation

                return None

            # Map columns for assignments
            assignments_col_mapping = {}
            missing_assignments_cols = []

            for col in required_assignments_cols:
                matched_col = find_column_match(assignments_df, col)
                if matched_col:
                    assignments_col_mapping[col] = matched_col
                else:
                    missing_assignments_cols.append(col)

            # Map columns for ZIP database
            zip_db_col_mapping = {}
            missing_zip_db_cols = []

            for col in required_zip_db_cols:
                matched_col = find_column_match(zip_db_df, col)
                if matched_col:
                    zip_db_col_mapping[col] = matched_col
                else:
                    missing_zip_db_cols.append(col)

            # Rename columns to standard names
            if not missing_assignments_cols:
                assignments_df = assignments_df.rename(columns=assignments_col_mapping)

            if not missing_zip_db_cols:
                zip_db_df = zip_db_df.rename(columns=zip_db_col_mapping)

            if missing_assignments_cols or missing_zip_db_cols:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Could not automatically match all required columns")

                if missing_assignments_cols:
                    st.write(f"**Assignments file - couldn't find:** {missing_assignments_cols}")
                    st.write("**Available columns:** ", list(assignments_df.columns))

                    # Manual column mapping for assignments
                    st.subheader("📝 Manual Column Mapping - Assignments File")
                    manual_assignments_mapping = {}

                    for missing_col in missing_assignments_cols:
                        selected_col = st.selectbox(
                            f"Select column for '{missing_col}':",
                            options=["-- Select --"] + list(assignments_df.columns),
                            key=f"assign_{missing_col}"
                        )
                        if selected_col != "-- Select --":
                            manual_assignments_mapping[missing_col] = selected_col

                    # Apply manual mapping
                    if len(manual_assignments_mapping) == len(missing_assignments_cols):
                        assignments_df = assignments_df.rename(columns=manual_assignments_mapping)
                        missing_assignments_cols = []
                        st.success("✅ Assignments file column mapping applied!")

                if missing_zip_db_cols:
                    st.write(f"**ZIP database - couldn't find:** {missing_zip_db_cols}")
                    st.write("**Available columns:** ", list(zip_db_df.columns))

                    # Manual column mapping for ZIP database
                    st.subheader("📝 Manual Column Mapping - ZIP Database")
                    manual_zip_mapping = {}

                    for missing_col in missing_zip_db_cols:
                        selected_col = st.selectbox(
                            f"Select column for '{missing_col}':",
                            options=["-- Select --"] + list(zip_db_df.columns),
                            key=f"zip_{missing_col}"
                        )
                        if selected_col != "-- Select --":
                            manual_zip_mapping[missing_col] = selected_col

                    # Apply manual mapping
                    if len(manual_zip_mapping) == len(missing_zip_db_cols):
                        zip_db_df = zip_db_df.rename(columns=manual_zip_mapping)
                        missing_zip_db_cols = []
                        st.success("✅ ZIP database column mapping applied!")

                # If still missing columns after manual mapping, show error and stop
                if missing_assignments_cols or missing_zip_db_cols:
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.stop()
                else:
                    st.markdown('</div>', unsafe_allow_html=True)

            # Display data overview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Assignments Data Overview")
                st.write(f"**Total records:** {len(assignments_df):,}")
                st.write(f"**Unique businesses:** {assignments_df['business_id'].nunique():,}")
                st.write(f"**Unique business ZIPs:** {assignments_df['biz_zip'].nunique():,}")
                st.write(f"**Unique target ZIPs:** {assignments_df['target_zip'].nunique():,}")

                with st.expander("View sample data"):
                    st.dataframe(assignments_df.head())

            with col2:
                st.subheader("🌍 ZIP Database Overview")
                st.write(f"**Total ZIP codes:** {len(zip_db_df):,}")
                st.write(f"**Latitude range:** {zip_db_df['lat'].min():.2f} to {zip_db_df['lat'].max():.2f}")
                st.write(f"**Longitude range:** {zip_db_df['lng'].min():.2f} to {zip_db_df['lng'].max():.2f}")

                with st.expander("View sample data"):
                    st.dataframe(zip_db_df.head())

            # Process button
            if st.button("🚀 Generate Polygons", type="primary", use_container_width=True):

                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                start_time = time.time()

                try:
                    # Run the polygon generation
                    results = process_polygon_generation(
                        assignments_df,
                        zip_db_df,
                        progress_bar,
                        status_text
                    )

                    end_time = time.time()
                    processing_time = round(end_time - start_time, 2)

                    # Display results
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Processing completed in {processing_time} seconds!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Calculate statistics
                    valid_polygons = sum(1 for r in results if r['has_valid_polygon'] == "Yes")
                    total_businesses = len(results)

                    if valid_polygons > 0:
                        avg_coverage = sum(r['coverage_percentage'] for r in results if
                                           r['has_valid_polygon'] == "Yes") / valid_polygons
                        best_coverage = max(
                            r['coverage_percentage'] for r in results if r['has_valid_polygon'] == "Yes")
                    else:
                        avg_coverage = 0
                        best_coverage = 0

                    # Display summary statistics
                    st.subheader("📈 Results Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Businesses", total_businesses)
                    with col2:
                        st.metric("Valid Polygons", valid_polygons)
                    with col3:
                        st.metric("Average Coverage", f"{avg_coverage:.1f}%")
                    with col4:
                        st.metric("Best Coverage", f"{best_coverage:.1f}%")

                    # Create results DataFrame
                    results_df = pd.DataFrame([
                        {
                            'business_id': r['business_id'],
                            'biz_zip': r['biz_zip'],
                            'boundary_zips': r['boundary_zips'],
                            'contained_zip_count': r['contained_zip_count'],
                            'total_assigned_zips': r['total_assigned_zips'],
                            'within_distance_constraint': r['within_distance_constraint'],
                            'coverage_percentage': r['coverage_percentage'],
                            'max_polygon_distance': r.get('max_polygon_distance', 0),
                            'has_valid_polygon': r['has_valid_polygon'],
                            'polygon_point_count': r.get('polygon_point_count', 0)
                        }
                        for r in results
                    ])

                    # Sort by coverage percentage
                    results_df = results_df.sort_values('coverage_percentage', ascending=False)

                    # Display top results
                    st.subheader("🏆 Top Performing Businesses")
                    top_results = results_df[results_df['has_valid_polygon'] == 'Yes'].head(10)
                    st.dataframe(
                        top_results,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Create covered ZIPs flattened CSV
                    covered_zips_flat = []
                    uncovered_zips_flat = []

                    try:
                        for r in results:
                            business_id = r['business_id']
                            biz_zip = r['biz_zip']

                            # Get all assigned ZIPs for this business from original data
                            business_assignments = assignments_df[assignments_df['business_id'] == business_id]
                            all_assigned_zips = business_assignments['target_zip'].tolist()

                            # Get contained ZIPs
                            contained_zips = r.get('contained_zips', [])
                            if not isinstance(contained_zips, list):
                                contained_zips = []

                            # Create flattened records for covered ZIPs
                            for zip_code in contained_zips:
                                covered_zips_flat.append({
                                    'business_id': business_id,
                                    'biz_zip': biz_zip,
                                    'target_zip': zip_code
                                })

                            # Create flattened records for uncovered ZIPs
                            uncovered_zips = [z for z in all_assigned_zips if z not in contained_zips]
                            for zip_code in uncovered_zips:
                                # Determine reason for not being covered
                                if r.get('has_valid_polygon', 'No') == "No":
                                    reason = r.get('reason', 'No valid polygon created')
                                else:
                                    reason = "Outside polygon boundary"

                                uncovered_zips_flat.append({
                                    'business_id': business_id,
                                    'biz_zip': biz_zip,
                                    'target_zip': zip_code,
                                    'reason': reason
                                })

                        # Create DataFrames
                        covered_df = pd.DataFrame(covered_zips_flat)
                        uncovered_df = pd.DataFrame(uncovered_zips_flat)

                        log_debug(f"Created {len(covered_zips_flat)} covered ZIP records")
                        log_debug(f"Created {len(uncovered_zips_flat)} uncovered ZIP records")

                    except Exception as e:
                        st.error(f"Error processing ZIP data: {str(e)}")
                        if debug_mode:
                            st.exception(e)
                        covered_df = pd.DataFrame()
                        uncovered_df = pd.DataFrame()

                    # Display additional statistics
                    st.subheader("📊 Coverage Breakdown")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Covered ZIPs", len(covered_zips_flat))
                    with col2:
                        st.metric("Total Uncovered ZIPs", len(uncovered_zips_flat))
                    with col3:
                        total_zips = len(covered_zips_flat) + len(uncovered_zips_flat)
                        coverage_rate = (len(covered_zips_flat) / total_zips * 100) if total_zips > 0 else 0
                        st.metric("Overall Coverage Rate", f"{coverage_rate:.1f}%")

                    # Store results in session state to prevent reprocessing
                    st.session_state['results_df'] = results_df
                    st.session_state['covered_df'] = covered_df
                    st.session_state['uncovered_df'] = uncovered_df
                    st.session_state['contained_zips_data'] = [
                        {
                            'business_id': r['business_id'],
                            'biz_zip': r['biz_zip'],
                            'contained_zips': r.get('contained_zips', []),
                            'polygon_point_count': r.get('polygon_point_count', 0)
                        }
                        for r in results
                    ]

                    # Download section
                    st.subheader("💾 Download Results")
                    st.info("💡 Files will download automatically when you click the buttons below")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Main results CSV
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Main Results CSV",
                            data=csv_data,
                            file_name="zip_code_polygons.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="main_results_download"
                        )

                    with col2:
                        # Covered ZIPs flattened CSV
                        if len(covered_zips_flat) > 0:
                            covered_csv = covered_df.to_csv(index=False)
                            st.download_button(
                                label="✅ Download Covered ZIPs CSV",
                                data=covered_csv,
                                file_name="covered_zips_flattened.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="covered_zips_download"
                            )
                        else:
                            st.info("No covered ZIPs to download")

                    with col3:
                        # Uncovered ZIPs CSV
                        if len(uncovered_zips_flat) > 0:
                            uncovered_csv = uncovered_df.to_csv(index=False)
                            st.download_button(
                                label="❌ Download Uncovered ZIPs CSV",
                                data=uncovered_csv,
                                file_name="uncovered_zips_flattened.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="uncovered_zips_download"
                            )
                        else:
                            st.info("No uncovered ZIPs to download")

                    # Additional download options
                    col1, col2 = st.columns(2)

                    with col1:
                        # Contained ZIPs JSON
                        json_data = json.dumps(st.session_state['contained_zips_data'], indent=2)
                        st.download_button(
                            label="📋 Download Contained ZIPs JSON",
                            data=json_data,
                            file_name="contained_zips_by_business.json",
                            mime="application/json",
                            use_container_width=True,
                            key="json_download"
                        )

                    with col2:
                        # Create visualization if possible
                        if FOLIUM_AVAILABLE:
                            if st.button("🗺️ Generate Map Visualization", use_container_width=True, key="generate_map"):
                                try:
                                    # Create visualization
                                    html_content = create_folium_visualization(
                                        results_df,
                                        zip_db_df,
                                        assignments_df,
                                        covered_df,
                                        uncovered_df
                                    )

                                    # Store in session state
                                    st.session_state['map_html'] = html_content

                                    st.success("✅ Map visualization generated!")

                                except Exception as e:
                                    st.error(f"❌ Error creating visualization: {str(e)}")
                                    if debug_mode:
                                        st.exception(e)
                        else:
                            st.info("🗺️ Map visualization unavailable")
                            st.caption("Install Folium to enable: pip install folium")

                    # Download map if available
                    if 'map_html' in st.session_state:
                        st.download_button(
                            label="🗺️ Download Interactive Map HTML",
                            data=st.session_state['map_html'],
                            file_name="polygon_visualization.html",
                            mime="text/html",
                            use_container_width=True,
                            key="map_download"
                        )
                        st.info("💡 Download the HTML file and open it in your browser to view the interactive map.")

                    # Show sample data from covered and uncovered ZIPs
                    if len(covered_zips_flat) > 0:
                        with st.expander("📋 Sample Covered ZIPs Data"):
                            st.dataframe(covered_df.head(10), use_container_width=True)

                    if len(uncovered_zips_flat) > 0:
                        with st.expander("📋 Sample Uncovered ZIPs Data"):
                            st.dataframe(uncovered_df.head(10), use_container_width=True)

                    # Show businesses without valid polygons
                    invalid_results = results_df[results_df['has_valid_polygon'] == 'No']
                    if len(invalid_results) > 0:
                        st.subheader(f"⚠️ Businesses Without Valid Polygons ({len(invalid_results)})")

                        # Group by reason
                        if 'reason' in invalid_results.columns:
                            reason_counts = invalid_results.groupby('reason').size().reset_index(name='count')
                        else:
                            reason_counts = pd.DataFrame({'reason': ['Unknown'], 'count': [len(invalid_results)]})

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            display_cols = ['business_id', 'biz_zip', 'total_assigned_zips',
                                            'within_distance_constraint']
                            if 'reason' in invalid_results.columns:
                                display_cols.append('reason')
                            st.dataframe(invalid_results[display_cols],
                                         use_container_width=True, hide_index=True)

                        with col2:
                            st.subheader("📊 Failure Reasons")
                            for _, row in reason_counts.iterrows():
                                st.write(f"**{row['reason']}:** {row['count']} businesses")

                        # Download businesses without valid polygons
                        invalid_csv = invalid_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Businesses Without Polygons",
                            data=invalid_csv,
                            file_name="businesses_without_polygons.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="invalid_businesses_download"
                        )

                    # Show full results
                    with st.expander("📋 View All Results"):
                        st.dataframe(results_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ An error occurred during processing: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    if debug_mode:
                        st.exception(e)

        except Exception as e:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error(f"❌ Error loading data files: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Instructions when no files are uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("👆 Please upload both required CSV files in the sidebar to get started.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📋 Required File Formats")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ZIP Code Assignments CSV:**")
            st.code("""
business_id,biz_zip,target_zip,distance_miles
1001,12345,12346,15.2
1001,12345,12347,22.8
1002,54321,54322,8.5
            """)

        with col2:
            st.markdown("**ZIP Code Database CSV:**")
            st.code("""
zip,lat,lng
12345,40.7128,-74.0060
12346,40.7589,-73.9851
12347,40.6892,-74.0445
            """)

        st.subheader("🔧 Algorithm Details")
        st.markdown("""
        - **Algorithm:** Graham Scan Convex Hull with distance constraints
        - **Distance Constraint:** Maximum 85 miles between any two polygon points
        - **Point Limit:** Maximum 6 points per polygon for optimal performance
        - **Coverage Calculation:** Percentage of assigned ZIPs contained within the generated polygon
        """)


# Prevent execution when running as script directly
if __name__ == "__main__":
    if is_streamlit_context():
        main()
    else:
        print("❌ This script must be run with Streamlit!")
        print("✅ Run: streamlit run streamlit_polygon_generation.py")
        print("📖 Make sure you have installed: pip install streamlit shapely geopy pandas numpy")