import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import math
import tempfile
import zipfile
from io import BytesIO
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
from sklearn.cluster import KMeans, DBSCAN
import warnings
import folium
import streamlit.components.v1 as components
import base64

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Business Coverage Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

def log_debug(message):
    """Helper function for debug logging"""
    st.write(f"DEBUG: {message}")

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

def improved_clustering(uncovered_zips_with_coords, zip_coordinates, min_cluster_size=3, iteration=1):
    """Much more aggressive clustering that creates smaller, more efficient service areas."""
    st.write(f"   🔍 Analyzing geographic distribution (iteration {iteration})...")

    # Prepare coordinates
    coords = np.array([[zip_coordinates[z][0], zip_coordinates[z][1]] for z in uncovered_zips_with_coords])

    # More aggressive clustering approach - favor smaller, tighter clusters
    best_clustering = None
    best_score = -1
    best_method = ""

    # Method 1: Much tighter DBSCAN - create smaller, more localized clusters
    distance_thresholds = [0.2, 0.3, 0.4, 0.5] if iteration <= 2 else [0.15, 0.25, 0.35]

    for eps in distance_thresholds:
        clustering = DBSCAN(eps=eps, min_samples=max(2, min_cluster_size - 1)).fit(coords)
        labels = clustering.labels_

        # Score heavily favors more clusters with reasonable sizes
        noise_count = sum(1 for l in labels if l == -1)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

        if cluster_count > 0:
            cluster_sizes = []
            for cluster_id in set(labels):
                if cluster_id != -1:
                    cluster_size = sum(1 for l in labels if l == cluster_id)
                    cluster_sizes.append(cluster_size)

            # Favor more clusters with sizes between 3-15 ZIPs
            optimal_size_clusters = sum(1 for size in cluster_sizes if 3 <= size <= 15)
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

            # Score: favor many small-medium clusters, penalize huge clusters
            score = (
                    cluster_count * 2.0 +  # More clusters is better
                    optimal_size_clusters * 1.5 +  # Clusters of good size
                    (noise_count * -0.3) +  # Some noise is OK
                    (1.0 / (1 + max(cluster_sizes) / 20) if cluster_sizes else 0)  # Penalize huge clusters
            )

            if score > best_score:
                best_score = score
                best_clustering = labels
                best_method = f"Tight-DBSCAN(eps={eps}, clusters={cluster_count})"

    # Method 2: Aggressive K-means with higher K values
    max_k = min(len(uncovered_zips_with_coords) // 2, 25)  # More clusters
    for k in range(max(3, len(uncovered_zips_with_coords) // 10), max_k):
        try:
            clustering = KMeans(n_clusters=k, random_state=42).fit(coords)
            labels = clustering.labels_
            cluster_centers = clustering.cluster_centers_

            # Calculate average distance from points to their cluster centers
            total_distance = 0
            cluster_sizes = []
            for i, label in enumerate(labels):
                center = cluster_centers[label]
                point = coords[i]
                total_distance += geodesic(center, point).miles

            # Count cluster sizes
            for cluster_id in range(k):
                size = sum(1 for l in labels if l == cluster_id)
                cluster_sizes.append(size)

            avg_distance = total_distance / len(labels)
            optimal_size_clusters = sum(1 for size in cluster_sizes if 3 <= size <= 12)

            # Score: favor compact clusters of good size
            score = (
                    k * 0.8 +  # More clusters
                    optimal_size_clusters * 1.2 +  # Good-sized clusters
                    (100.0 / (1 + avg_distance))  # Compact clusters (lower distance)
            )

            if score > best_score:
                best_score = score
                best_clustering = labels
                best_method = f"Aggressive-KMeans(k={k}, avg_dist={avg_distance:.1f}mi)"

        except:
            continue

    st.write(f"   ✓ Best method: {best_method} (score: {best_score:.2f})")

    # Group ZIPs by cluster
    clusters = {}
    noise_zips = []

    for i, zip_code in enumerate(uncovered_zips_with_coords):
        cluster_id = best_clustering[i] if best_clustering is not None else 0
        if cluster_id == -1:  # Noise/outlier (only for DBSCAN)
            noise_zips.append(zip_code)
        else:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(zip_code)

    # Further split any overly large clusters (>20 ZIPs)
    final_clusters = {}
    cluster_counter = 0

    for cluster_id, zip_list in clusters.items():
        if len(zip_list) > 20:
            # Split large cluster into smaller sub-clusters
            sub_coords = np.array([[zip_coordinates[z][0], zip_coordinates[z][1]] for z in zip_list])
            n_subclusters = min(len(zip_list) // 8, 4)  # Split into smaller pieces

            try:
                sub_clustering = KMeans(n_clusters=n_subclusters, random_state=42).fit(sub_coords)
                sub_labels = sub_clustering.labels_

                for sub_id in range(n_subclusters):
                    sub_zips = [zip_list[i] for i, label in enumerate(sub_labels) if label == sub_id]
                    if len(sub_zips) >= 3:
                        final_clusters[cluster_counter] = sub_zips
                        cluster_counter += 1

            except:
                # If sub-clustering fails, keep original
                final_clusters[cluster_counter] = zip_list
                cluster_counter += 1
        else:
            final_clusters[cluster_counter] = zip_list
            cluster_counter += 1

    return final_clusters, noise_zips, best_method

def create_coverage_analysis(zip_db_df, uncovered_df, covered_df=None):
    """Main function to create coverage analysis with progress tracking"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading data...")
    progress_bar.progress(10)
    
    # Process data
    zip_coordinates = {int(row['zip']): (row['lat'], row['lng']) for _, row in zip_db_df.iterrows()}
    
    covered_zips = set()
    used_business_zips = set()
    if covered_df is not None:
        covered_zips = set(covered_df['zip_code'].unique())
        used_business_zips = set(covered_df['biz_zip'].unique())
    
    uncovered_zips = list(uncovered_df['zip_code'].unique())
    uncovered_zips_with_coords = [z for z in uncovered_zips if z in zip_coordinates]
    
    status_text.text(f"Processing {len(uncovered_zips_with_coords)} uncovered ZIPs...")
    progress_bar.progress(20)
    
    if len(uncovered_zips_with_coords) == 0:
        st.success("No uncovered ZIPs to process!")
        return None
    
    # Initialize helper functions
    distance_cache = {}

    def cached_distance(zip1, zip2):
        cache_key = (min(zip1, zip2), max(zip1, zip2))
        if cache_key in distance_cache:
            return distance_cache[cache_key]

        if zip1 not in zip_coordinates or zip2 not in zip_coordinates:
            return float('inf')

        coords1 = zip_coordinates[zip1]
        coords2 = zip_coordinates[zip2]
        distance = geodesic(coords1, coords2).miles
        distance_cache[cache_key] = distance
        return distance

    def create_constrained_convex_hull(business_zip, target_zips, max_distance=85, max_points=6):
        """Create constrained convex hull"""
        if len(target_zips) < 3:
            return [], []

        # Filter by distance constraint
        valid_zips = []
        for zip_code in target_zips:
            if zip_code in zip_coordinates and business_zip in zip_coordinates:
                distance = cached_distance(business_zip, zip_code)
                if distance <= max_distance:
                    valid_zips.append(zip_code)

        if len(valid_zips) < 3:
            return [], []

        # Create distance-constrained clusters
        clusters = []
        for center_zip in valid_zips[:min(30, len(valid_zips))]:
            cluster = [center_zip]
            for other_zip in valid_zips:
                if other_zip == center_zip:
                    continue

                valid_for_cluster = True
                for cluster_zip in cluster:
                    if cached_distance(other_zip, cluster_zip) > max_distance:
                        valid_for_cluster = False
                        break

                if valid_for_cluster:
                    cluster.append(other_zip)

            if len(cluster) >= 3:
                clusters.append(cluster)

        if not clusters:
            return [], []

        # Find best convex hull
        best_hull_zips = []
        best_contained_zips = []
        best_coverage = 0

        for cluster in clusters[:3]:  # Top 3 clusters
            cluster_points = []
            zip_to_point = {}
            for zip_code in cluster:
                if zip_code in zip_coordinates:
                    lat, lng = zip_coordinates[zip_code]
                    point = (lng, lat)
                    cluster_points.append(point)
                    zip_to_point[point] = zip_code

            if len(cluster_points) < 3:
                continue

            hull_points = graham_scan(cluster_points)

            # Limit to max_points
            if len(hull_points) > max_points:
                from itertools import combinations
                if len(hull_points) > 12:
                    step = len(hull_points) // max_points
                    hull_subset = [hull_points[i] for i in range(0, len(hull_points), step)][:max_points]
                    hull_combinations = [hull_subset]
                else:
                    hull_combinations = list(combinations(hull_points, max_points))[:10]

                best_subset_coverage = 0
                best_subset_hull = []
                best_subset_contained = []

                for hull_subset in hull_combinations:
                    subset_hull_zips = [zip_to_point[point] for point in hull_subset]

                    # Validate distance constraints
                    valid = True
                    for i in range(len(subset_hull_zips)):
                        for j in range(i + 1, len(subset_hull_zips)):
                            if cached_distance(subset_hull_zips[i], subset_hull_zips[j]) > max_distance:
                                valid = False
                                break
                        if not valid:
                            break

                    if not valid:
                        continue

                    # Count contained ZIPs
                    coverage_count, contained = count_contained_zips(subset_hull_zips, valid_zips)
                    if coverage_count > best_subset_coverage:
                        best_subset_coverage = coverage_count
                        best_subset_hull = subset_hull_zips
                        best_subset_contained = contained

                hull_zips = best_subset_hull
                coverage_count = best_subset_coverage
                contained = best_subset_contained
            else:
                hull_zips = [zip_to_point[point] for point in hull_points]
                coverage_count, contained = count_contained_zips(hull_zips, valid_zips)

            if coverage_count > best_coverage:
                best_coverage = coverage_count
                best_hull_zips = hull_zips
                best_contained_zips = contained

        return best_hull_zips, best_contained_zips

    def count_contained_zips(polygon_zips, target_zips):
        """Count contained ZIPs using Shapely"""
        if len(polygon_zips) < 3:
            return 0, []

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
                try:
                    from shapely.validation import make_valid
                    polygon = make_valid(polygon)
                except ImportError:
                    polygon_points.reverse()
                    polygon = Polygon(polygon_points)
        except Exception:
            return 0, []

        contained_zips = []
        minx, miny, maxx, maxy = polygon.bounds

        for zip_code in target_zips:
            if zip_code not in zip_coordinates:
                continue
            lng, lat = zip_coordinates[zip_code][1], zip_coordinates[zip_code][0]
            if minx <= lng <= maxx and miny <= lat <= maxy:
                point = Point(lng, lat)
                if polygon.contains(point) or polygon.touches(point):
                    contained_zips.append(zip_code)

        return len(contained_zips), contained_zips

    # Start iterative coverage process
    status_text.text("Starting iterative coverage analysis...")
    progress_bar.progress(30)
    
    all_new_locations = []
    total_newly_covered = set()
    current_uncovered = set(uncovered_zips_with_coords)
    iteration = 0
    max_iterations = 5

    while len(current_uncovered) > 0 and iteration < max_iterations:
        iteration += 1
        st.write(f"\n   🔄 ITERATION {iteration}: Processing {len(current_uncovered)} uncovered ZIPs...")
        
        progress_value = 30 + (iteration * 12)
        progress_bar.progress(progress_value)
        status_text.text(f"Iteration {iteration}: Clustering and analyzing...")

        # Clustering
        clusters, noise_zips, method = improved_clustering(
            list(current_uncovered),
            zip_coordinates,
            min_cluster_size=max(2, 6 - iteration),
            iteration=iteration
        )

        st.write(f"      📊 {method} found {len(clusters)} clusters + {len(noise_zips)} outliers")

        iteration_new_locations = []
        iteration_covered = set()

        # Process clusters
        for cluster_id, target_zips in clusters.items():
            if len(target_zips) < 2:
                continue

            st.write(f"      🎯 Processing Cluster {cluster_id}: {len(target_zips)} ZIPs")

            # Find optimal business ZIP
            lats = [zip_coordinates[z][0] for z in target_zips if z in zip_coordinates]
            lngs = [zip_coordinates[z][1] for z in target_zips if z in zip_coordinates]

            if not lats:
                continue

            centroid_lat = sum(lats) / len(lats)
            centroid_lng = sum(lngs) / len(lngs)

            # Expanded candidate search
            candidate_zips = []
            max_search_distance = 50 + (iteration * 20)

            for zip_code, (lat, lng) in zip_coordinates.items():
                if zip_code not in used_business_zips and zip_code not in covered_zips:
                    distance_to_centroid = geodesic((centroid_lat, centroid_lng), (lat, lng)).miles
                    if distance_to_centroid <= max_search_distance:
                        candidate_zips.append((zip_code, distance_to_centroid))

            candidate_zips.sort(key=lambda x: x[1])
            test_count = min(20 + (iteration * 10), len(candidate_zips))

            best_business_zip = None
            best_coverage = 0
            best_polygon = []
            best_contained = []

            for business_zip, distance_to_centroid in candidate_zips[:test_count]:
                polygon, contained = create_constrained_convex_hull(business_zip, target_zips)
                coverage = len(contained)

                min_acceptable_coverage = max(1, len(target_zips) // (3 + iteration))

                if coverage >= min_acceptable_coverage and coverage > best_coverage:
                    best_coverage = coverage
                    best_business_zip = business_zip
                    best_polygon = polygon
                    best_contained = contained

            if best_business_zip and best_coverage > 0:
                # Calculate stats
                max_distance = 0
                for i in range(len(best_polygon)):
                    for j in range(i + 1, len(best_polygon)):
                        if best_polygon[i] in zip_coordinates and best_polygon[j] in zip_coordinates:
                            dist = cached_distance(best_polygon[i], best_polygon[j])
                            max_distance = max(max_distance, dist)

                coverage_pct = round((len(best_contained) / len(target_zips)) * 100)

                location_data = {
                    'new_business_id': f"NEW_BIZ_{len(all_new_locations) + len(iteration_new_locations) + 1:03d}",
                    'business_zip': best_business_zip,
                    'boundary_zips': ';'.join(map(str, best_polygon)),
                    'contained_zip_count': len(best_contained),
                    'contained_zips': best_contained,
                    'target_zip_count': len(target_zips),
                    'coverage_percentage': coverage_pct,
                    'polygon_point_count': len(best_polygon),
                    'max_polygon_distance': round(max_distance, 2),
                    'has_valid_polygon': "Yes",
                    'iteration': iteration
                }

                iteration_new_locations.append(location_data)
                iteration_covered.update(best_contained)
                used_business_zips.add(best_business_zip)

                st.write(f"         ✅ Business ZIP {best_business_zip}: {len(best_contained)}/{len(target_zips)} ZIPs ({coverage_pct}%)")

        # Update for next iteration
        all_new_locations.extend(iteration_new_locations)
        total_newly_covered.update(iteration_covered)
        current_uncovered = current_uncovered - iteration_covered

        st.write(f"      📈 Iteration {iteration} results:")
        st.write(f"         - New locations: {len(iteration_new_locations)}")
        st.write(f"         - ZIPs covered: {len(iteration_covered)}")
        st.write(f"         - Remaining uncovered: {len(current_uncovered)}")

        # Stop if no progress
        min_progress = max(1, len(uncovered_zips_with_coords) // 50)
        if len(iteration_covered) < min_progress and iteration > 2:
            st.write(f"      ⏹️  Minimal progress in iteration {iteration}, stopping early")
            break

    progress_bar.progress(90)
    status_text.text("Generating results...")

    # Generate results
    if not all_new_locations:
        st.error("No new business locations created")
        return None

    coverage_rate = (len(total_newly_covered) / len(uncovered_zips_with_coords)) * 100

    results = {
        'all_new_locations': all_new_locations,
        'total_newly_covered': total_newly_covered,
        'current_uncovered': current_uncovered,
        'coverage_rate': coverage_rate,
        'iteration': iteration,
        'zip_coordinates': zip_coordinates
    }

    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    return results

def create_folium_map(results):
    """Create enhanced Folium map for visualization"""
    if not results:
        return None
    
    all_new_locations = results['all_new_locations']
    total_newly_covered = results['total_newly_covered']
    current_uncovered = results['current_uncovered']
    coverage_rate = results['coverage_rate']
    zip_coordinates = results['zip_coordinates']
    
    # Calculate map center
    all_coords = list(zip_coordinates.values())
    mean_lat = sum([c[0] for c in all_coords]) / len(all_coords)
    mean_lng = sum([c[1] for c in all_coords]) / len(all_coords)
    
    # Create map
    m = folium.Map(location=[mean_lat, mean_lng], zoom_start=4, tiles='cartodbpositron')
    
    # Calculate stats for reference (but don't display overlay)
    total_new_covered = len(total_newly_covered)
    total_new_locations = len(all_new_locations)
    still_uncovered_count = len(current_uncovered)

    # Add NEW business locations and coverage
    for location in all_new_locations:
        business_zip = int(location['business_zip'])
        new_business_id = str(location['new_business_id'])

        # Parse boundary ZIPs
        boundary_zips = []
        if isinstance(location['boundary_zips'], str) and location['boundary_zips']:
            boundary_zips = [int(z) for z in str(location['boundary_zips']).split(';') if z.strip().isdigit()]

        # Parse contained ZIPs
        contained_zips = location['contained_zips']

        # Draw NEW service area polygon (bright orange)
        poly_points = [zip_coordinates[z] for z in boundary_zips if z in zip_coordinates]
        if len(poly_points) >= 3:
            folium.Polygon(
                locations=poly_points,
                color='darkorange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.4,
                weight=4,
                tooltip=f"🆕 {new_business_id} Service Area",
                popup=folium.Popup(f'''
                    <div style="font-size: 14px;">
                        <b>🆕 {new_business_id}</b><br>
                        <b>Business ZIP:</b> {business_zip}<br>
                        <b>Service Area:</b> {len(boundary_zips)} boundary points<br>
                        <b>Covers:</b> {len(contained_zips)} ZIPs<br>
                        <b>Coverage:</b> {location.get('coverage_percentage', 0)}%
                    </div>
                ''', max_width=300)
            ).add_to(m)

        # Draw NEW business location (bright gold star)
        if business_zip in zip_coordinates:
            folium.Marker(
                location=zip_coordinates[business_zip],
                icon=folium.Icon(color='orange', icon='star', prefix='fa'),
                tooltip=f"🆕 {new_business_id}",
                popup=folium.Popup(f'''
                    <div style="font-size: 14px;">
                        <b>🆕 NEW BUSINESS LOCATION</b><br>
                        <b>ID:</b> {new_business_id}<br>
                        <b>ZIP:</b> {business_zip}<br>
                        <b>Coverage:</b> {len(contained_zips)} ZIPs ({location.get('coverage_percentage', 0)}%)<br>
                        <b>Service Area:</b> {len(boundary_zips)} boundary points<br>
                        <b>Max Distance:</b> {location.get('max_polygon_distance', 0)} miles
                    </div>
                ''', max_width=350)
            ).add_to(m)

    # Draw NEWLY covered ZIPs (bright green dots)
    for zip_code in total_newly_covered:
        if zip_code in zip_coordinates:
            folium.CircleMarker(
                location=zip_coordinates[zip_code],
                radius=6,
                color='darkgreen',
                fill=True,
                fill_color='lime',
                fill_opacity=0.9,
                weight=2,
                tooltip=f"🆕 NEWLY Covered: {zip_code}"
            ).add_to(m)

    # Draw STILL uncovered ZIPs (red X markers)
    for zip_code in current_uncovered:
        if zip_code in zip_coordinates:
            folium.Marker(
                location=zip_coordinates[zip_code],
                icon=folium.DivIcon(
                    html='<div style="color:red;font-size:18px;font-weight:bold;text-shadow:1px 1px 2px white;">❌</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                ),
                tooltip=f"❌ Still Uncovered: {zip_code}"
            ).add_to(m)

    return m

def create_downloadable_files(results):
    """Create all downloadable files"""
    if not results:
        return None
    
    all_new_locations = results['all_new_locations']
    total_newly_covered = results['total_newly_covered']
    current_uncovered = results['current_uncovered']
    
    files = {}
    
    # 1. New business locations CSV
    results_df = pd.DataFrame([
        {
            'new_business_id': r['new_business_id'],
            'business_zip': r['business_zip'],
            'boundary_zips': r['boundary_zips'],
            'contained_zip_count': r['contained_zip_count'],
            'contained_zips': str(r['contained_zips']),
            'target_zip_count': r['target_zip_count'],
            'coverage_percentage': r['coverage_percentage'],
            'polygon_point_count': r['polygon_point_count'],
            'max_polygon_distance': r['max_polygon_distance'],
            'has_valid_polygon': r['has_valid_polygon'],
            'iteration': r['iteration']
        }
        for r in all_new_locations
    ])
    files['new_business_locations.csv'] = results_df.to_csv(index=False)
    
    # 2. Newly covered ZIPs CSV
    all_newly_covered_records = []
    for location in all_new_locations:
        for zip_code in location['contained_zips']:
            all_newly_covered_records.append({
                'new_business_id': location['new_business_id'],
                'business_zip': location['business_zip'],
                'zip_code': zip_code,
                'polygon_boundary': location['boundary_zips'],
                'iteration': location['iteration']
            })
    
    if all_newly_covered_records:
        newly_covered_df = pd.DataFrame(all_newly_covered_records)
        files['newly_covered_zips.csv'] = newly_covered_df.to_csv(index=False)
    
    # 3. JSON format
    json_results = [
        {
            'new_business_id': str(r['new_business_id']),
            'business_zip': int(r['business_zip']),
            'contained_zips': [int(z) for z in r['contained_zips']],
            'polygon_point_count': int(r['polygon_point_count']),
            'iteration': int(r['iteration'])
        }
        for r in all_new_locations
    ]
    files['new_business_locations.json'] = json.dumps(json_results, indent=2)
    
    # 4. Still uncovered ZIPs
    if current_uncovered:
        still_uncovered_df = pd.DataFrame([
            {'zip_code': zip_code, 'reason': 'Not covered after all iterations'}
            for zip_code in current_uncovered
        ])
        files['still_uncovered_zips.csv'] = still_uncovered_df.to_csv(index=False)
    
    return files

def create_zip_download(files, map_html=None):
    """Create a ZIP file containing all results"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all CSV and JSON files
        for filename, content in files.items():
            zip_file.writestr(filename, content)
        
        # Add map HTML if provided
        if map_html:
            zip_file.writestr('coverage_map.html', map_html)
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("🎯 Business Coverage Analysis Tool")
    st.markdown("Upload your data files to analyze uncovered ZIP codes and generate optimal business locations.")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Data Files")
        
        # File upload 1: ZIP Code Database
        zip_db_file = st.file_uploader(
            "Upload ZIP Code Database (CSV)",
            type=['csv'],
            help="CSV file containing ZIP codes with latitude and longitude coordinates"
        )
        
        # File upload 2: Uncovered ZIPs
        uncovered_file = st.file_uploader(
            "Upload Uncovered ZIPs (CSV)",
            type=['csv'],
            help="CSV file containing uncovered ZIP codes by business"
        )
        
        # Optional: Covered ZIPs (for reference)
        covered_file = st.file_uploader(
            "Upload Covered ZIPs (CSV) - Optional",
            type=['csv'],
            help="Optional: CSV file containing already covered ZIP codes"
        )
        
    # Main content area
    if zip_db_file is not None and uncovered_file is not None:
        
        # Load data
        try:
            zip_db_df = pd.read_csv(zip_db_file)
            uncovered_df = pd.read_csv(uncovered_file)
            covered_df = pd.read_csv(covered_file) if covered_file is not None else None
            
            st.success("✅ Data files loaded successfully!")
            
            # Display data summaries
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ZIP Coordinates", len(zip_db_df))
            
            with col2:
                st.metric("Uncovered ZIPs", len(uncovered_df['zip_code'].unique()))
            
            # Run analysis button
            if st.button("🚀 Run Coverage Analysis", type="primary"):
                with st.spinner("Running comprehensive coverage analysis..."):
                    results = create_coverage_analysis(zip_db_df, uncovered_df, covered_df)
                    
                    if results:
                        st.session_state.results_data = results
                        st.session_state.analysis_complete = True
                        st.success("🎉 Analysis completed successfully!")
                        st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading data files: {str(e)}")
            st.info("Please check that your CSV files have the correct format and column names.")
    
    else:
        st.info("👆 Please upload the required data files to begin analysis.")
        
        # Show example data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("**ZIP Code Database CSV should contain:**")
            st.code("zip,lat,lng\n12345,40.7128,-74.0060\n67890,34.0522,-118.2437")
            
            st.markdown("**Uncovered ZIPs CSV should contain:**")
            st.code("zip_code\n12345\n67890\n54321")
            
            st.markdown("**Covered ZIPs CSV should contain (optional):**")
            st.code("zip_code,biz_zip\n11111,22222\n33333,44444")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results_data:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        results = st.session_state.results_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("New Locations", len(results['all_new_locations']))
        
        with col2:
            st.metric("ZIPs Newly Covered", len(results['total_newly_covered']))
        
        with col3:
            st.metric("ZIPs Still Uncovered", len(results['current_uncovered']))
        
        with col4:
            st.metric("Coverage Rate", f"{results['coverage_rate']:.1f}%")
        
        # Create and display map
        st.subheader("🗺️ Interactive Coverage Map")
        
        folium_map = create_folium_map(results)
        if folium_map:
            # Display map using streamlit components instead of streamlit_folium
            map_html = folium_map._repr_html_()
            components.html(map_html, height=600, scrolling=True)
        else:
            map_html = None
        
        # Detailed results table
        st.subheader("📋 New Business Locations Details")
        
        locations_df = pd.DataFrame([
            {
                'Business ID': r['new_business_id'],
                'Business ZIP': r['business_zip'],
                'ZIPs Covered': r['contained_zip_count'],
                'Coverage %': f"{r['coverage_percentage']}%",
                'Service Area Points': r['polygon_point_count'],
                'Max Distance (mi)': r['max_polygon_distance'],
                'Iteration': r['iteration']
            }
            for r in results['all_new_locations']
        ])
        
        st.dataframe(locations_df, use_container_width=True)
        
        # Statistics by iteration
        if len(results['all_new_locations']) > 0:
            st.subheader("📈 Performance by Iteration")
            
            iteration_stats = {}
            for location in results['all_new_locations']:
                iter_num = location['iteration']
                if iter_num not in iteration_stats:
                    iteration_stats[iter_num] = {'locations': 0, 'zips_covered': 0}
                iteration_stats[iter_num]['locations'] += 1
                iteration_stats[iter_num]['zips_covered'] += len(location['contained_zips'])
            
            iter_df = pd.DataFrame([
                {
                    'Iteration': iter_num,
                    'New Locations': stats['locations'],
                    'ZIPs Covered': stats['zips_covered'],
                    'Avg ZIPs per Location': round(stats['zips_covered'] / stats['locations'], 1)
                }
                for iter_num, stats in iteration_stats.items()
            ])
            
            st.dataframe(iter_df, use_container_width=True)
        
        # Download section
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        # Create downloadable files
        files = create_downloadable_files(results)
        
        if files:
            # Individual file downloads
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Business Locations CSV",
                    data=files['new_business_locations.csv'],
                    file_name='new_business_locations.csv',
                    mime='text/csv'
                )
                
                if 'newly_covered_zips.csv' in files:
                    st.download_button(
                        label="📄 Download Newly Covered ZIPs CSV",
                        data=files['newly_covered_zips.csv'],
                        file_name='newly_covered_zips.csv',
                        mime='text/csv'
                    )
            
            with col2:
                st.download_button(
                    label="📄 Download JSON Results",
                    data=files['new_business_locations.json'],
                    file_name='new_business_locations.json',
                    mime='application/json'
                )
                
                if 'still_uncovered_zips.csv' in files:
                    st.download_button(
                        label="📄 Download Still Uncovered ZIPs",
                        data=files['still_uncovered_zips.csv'],
                        file_name='still_uncovered_zips.csv',
                        mime='text/csv'
                    )
            
            # ZIP download with all files
            st.markdown("**Download Everything:**")
            
            zip_data = create_zip_download(files, map_html)
            st.download_button(
                label="🗜️ Download All Results (ZIP)",
                data=zip_data.getvalue(),
                file_name='coverage_analysis_results.zip',
                mime='application/zip'
            )
            
            # Map HTML download
            if map_html:
                st.download_button(
                    label="🗺️ Download Interactive Map (HTML)",
                    data=map_html,
                    file_name='coverage_map.html',
                    mime='text/html'
                )
        
        # Success message and next steps
        st.markdown("---")
        st.success("🎉 Analysis Complete!")
        
        success_threshold = 80
        if results['coverage_rate'] >= success_threshold:
            st.balloons()
            st.success(f"🎯 **Excellent Coverage!** Achieved {results['coverage_rate']:.1f}% coverage rate.")
        else:
            st.info(f"📈 **Good Progress!** Achieved {results['coverage_rate']:.1f}% coverage rate.")
        
        if len(results['current_uncovered']) > 0:
            with st.expander("💡 Next Steps for Remaining ZIPs"):
                st.markdown(f"""
                **{len(results['current_uncovered'])} ZIPs still need coverage:**
                
                1. **Review the map** - Check if remaining ZIPs are in remote/isolated areas
                2. **Manual placement** - Consider strategic manual placement for outliers  
                3. **Adjust parameters** - Try different clustering parameters
                4. **Re-run analysis** - Upload updated data and run again
                
                The remaining uncovered ZIPs are available in the download files.
                """)

if __name__ == "__main__":
    main()
