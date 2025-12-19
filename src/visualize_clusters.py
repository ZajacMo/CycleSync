
import pandas as pd
import folium
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_clusters(
    clustered_data_path="data/output/clustered_data.csv",
    top_50_path="data/output/top_50_stations.csv",
    output_dir="data/output"
):
    # Configuration
    map_html_path = os.path.join(output_dir, "station_clusters_map.html")
    map_png_path = os.path.join(output_dir, "station_clusters_static.png")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    if not os.path.exists(clustered_data_path):
        print(f"Error: {clustered_data_path} not found.")
        return
        
    df = pd.read_csv(clustered_data_path)
    
    if os.path.exists(top_50_path):
        top_50_df = pd.read_csv(top_50_path)
        top_50_ids = set(top_50_df['station_id'].tolist())
        print(f"Loaded {len(top_50_ids)} Top 50 stations.")
    else:
        print(f"Warning: {top_50_path} not found. Proceeding without Top 50 distinction.")
        top_50_ids = set()

    # Calculate centroids and counts
    print("Calculating cluster centroids...")

    # Start points
    start_pts = df[['start_location_x', 'start_location_y', 'start_cluster']].copy()
    start_pts.columns = ['x', 'y', 'cluster']


    # End points
    end_pts = df[['end_location_x', 'end_location_y', 'end_cluster']].copy()
    end_pts.columns = ['x', 'y', 'cluster']

    # Combine
    all_pts = pd.concat([start_pts, end_pts], ignore_index=True)

    # Remove noise (-1)
    valid_pts = all_pts[all_pts['cluster'] != -1]

    # Calculate centroids and volume
    cluster_info = valid_pts.groupby('cluster').agg({
        'x': 'mean',
        'y': 'mean',
        'cluster': 'count'
    }).rename(columns={'cluster': 'volume'}).reset_index()

    # Add is_top50 flag
    cluster_info['is_top50'] = cluster_info['cluster'].apply(lambda x: x in top_50_ids)

    print(f"Found {len(cluster_info)} unique clusters.")

    # Split into two groups
    top50_clusters = cluster_info[cluster_info['is_top50']]
    other_clusters = cluster_info[~cluster_info['is_top50']]

    # --- 1. Static Plot with Matplotlib ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("Generating static map...")
    plt.figure(figsize=(12, 10))

    # Plot noise points (sample for performance)
    noise_pts = all_pts[all_pts['cluster'] == -1]
    if len(noise_pts) > 10000:
        noise_pts = noise_pts.sample(10000, random_state=42)
    plt.scatter(noise_pts['x'], noise_pts['y'], c='lightgray', s=1, alpha=0.3, label='噪声点')

    # Plot Other Stations (Blue)
    # Size proportional to log volume, but smaller
    if not other_clusters.empty:
        sizes_other = np.log1p(other_clusters['volume']) * 5
        plt.scatter(other_clusters['x'], other_clusters['y'], 
                    c='cornflowerblue', 
                    s=sizes_other, 
                    alpha=0.6, 
                    edgecolors='none', 
                    label='其他站点')

    # Plot Top 50 Stations (Red)
    if not top50_clusters.empty:
        # Larger size
        sizes_top = np.log1p(top50_clusters['volume']) * 15
        plt.scatter(top50_clusters['x'], top50_clusters['y'], 
                    c='red', 
                    s=sizes_top, 
                    alpha=0.8, 
                    edgecolors='black', 
                    linewidth=0.5,
                    label='Top50站点')

    plt.title('共享单车站点分布：Top50与其他')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(markerscale=1.5) # Scale up legend markers
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(map_png_path, dpi=150, bbox_inches='tight')
    print(f"Static map saved to {map_png_path}")

    # --- 2. Interactive Map with Folium ---
    print("Generating interactive map...")

    # Center map
    center_lat = cluster_info['y'].mean()
    center_lon = cluster_info['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')

    # Add Other Clusters (Blue circles)
    for idx, row in other_clusters.iterrows():
        cid = int(row['cluster'])
        vol = int(row['volume'])
        
        popup_text = f"<b>站点ID：</b> {cid}<br><b>流量：</b> {vol}<br>类型：普通"
        
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=2 + np.log1p(vol) * 0.5,
            popup=popup_text,
            color='cornflowerblue',
            fill=True,
            fill_color='cornflowerblue',
            fill_opacity=0.6,
            weight=1
        ).add_to(m)

    # Add Top 50 Clusters (Red circles with distinct style)
    for idx, row in top50_clusters.iterrows():
        cid = int(row['cluster'])
        vol = int(row['volume'])
        
        popup_text = f"<b>站点ID：</b> {cid}<br><b>流量：</b> {vol}<br>类型：<b>Top50</b>"
        
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=4 + np.log1p(vol) * 0.8, # Slightly larger
            popup=popup_text,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.8,
            weight=2 # Thicker border
        ).add_to(m)

    # Save
    m.save(map_html_path)
    print(f"Interactive map saved to {map_html_path}")

if __name__ == "__main__":
    visualize_clusters()
