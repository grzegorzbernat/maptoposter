import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    
    # Verify fonts exist
    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        # Fallback to embedded default theme
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    
    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme

# Load theme (can be changed via command line or input)
THEME = None  # Will be loaded later

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def get_edge_colors_by_type(G):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []
    
    for u, v, data in G.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')
        
        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign color based on road type
        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign width based on road importance
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

COORDS_CACHE_FILE = "coords_cache.json"

def load_coords_cache():
    """Load cached coordinates from file."""
    if os.path.exists(COORDS_CACHE_FILE):
        try:
            with open(COORDS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_coords_cache(cache):
    """Save coordinates cache to file."""
    with open(COORDS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Uses local cache to avoid repeated API calls.
    """
    cache_key = f"{city.lower()},{country.lower()}"
    cache = load_coords_cache()
    
    # Check cache first
    if cache_key in cache:
        coords = cache[cache_key]
        print(f"✓ Found in cache: {city}, {country}")
        print(f"✓ Coordinates: {coords[0]}, {coords[1]}")
        return tuple(coords)
    
    # Not in cache - fetch from Nominatim
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    location = geolocator.geocode(f"{city}, {country}")
    
    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        
        # Save to cache
        coords = (location.latitude, location.longitude)
        cache[cache_key] = coords
        save_coords_cache(cache)
        print("✓ Saved to cache")
        
        return coords
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")

def create_poster(city, country, point, dist, output_file, output_format='png', pois=None, draw_boundary=False, icon_size=50):
    print(f"\nGenerating map for {city}, {country}...")
    
    # Progress bar for data fetching
    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        pbar.update(1)
        time.sleep(0.5)  # Rate limit between requests
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
        except:
            water = None
        pbar.update(1)
        time.sleep(0.3)
        
        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(point, tags={'leisure': 'park', 'landuse': 'grass'}, dist=dist)
        except:
            parks = None
        pbar.update(1)
    
    print("✓ All data downloaded successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    # A4 = 8.27 x 11.69 inches, A3 = 11.69 x 16.54 inches
    if output_format.lower() == 'pdf':
        fig_size = (11.69, 16.54)  # A3 portrait for PDF
    else:
        fig_size = (8.27, 11.69)   # A4 portrait for PNG/SVG
    fig, ax = plt.subplots(figsize=fig_size, facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position([0, 0, 1, 1])
    
    # 3. Plot Layers
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    
    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)
    
    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)
    
    ox.plot_graph(
        G, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )
    
    # Get map extent for POI filtering
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox = (xlim[0], xlim[1], ylim[0], ylim[1])
    
    # Layer 3: Boundary (optional)
    if draw_boundary:
        print("Fetching and drawing city boundary...")
        try:
            # Fetch boundary polygon
            # Note: exact city/country string matching is important here
            boundary_gdf = ox.geocode_to_gdf(f"{city}, {country}")
            
            # Plot boundary
            # Using accent color or a default valid color if standard 'text' color is too similar to bg
            boundary_color = THEME.get('accent', THEME['text'])
            
            boundary_gdf.plot(ax=ax, facecolor='none', edgecolor=boundary_color, 
                              linewidth=1.5, linestyle='--', zorder=5)
            print("✓ Boundary drawn")
        except Exception as e:
            print(f"⚠ Could not fetch/draw boundary: {e}")

    # Layer 4: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)
    
    # 4. Typography using Roboto font
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_top = FontProperties(fname=FONTS['bold'], size=40)
        font_sub = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
    else:
        # Fallback to system fonts
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_top = FontProperties(family='monospace', weight='bold', size=40)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
    
    spaced_city = "  ".join(list(city.upper()))
    
    # Dynamically adjust font size based on city name length to prevent truncation
    base_font_size = 60
    city_char_count = len(city)
    if city_char_count > 10:
        # Scale down font size for longer names
        scale_factor = 10 / city_char_count
        adjusted_font_size = max(base_font_size * scale_factor, 24)  # Minimum size of 24
    else:
        adjusted_font_size = base_font_size
    
    if FONTS:
        font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

    # Translate country to Polish if needed
    COUNTRY_TRANSLATIONS = {
        "POLAND": "POLSKA",
        "GERMANY": "NIEMCY",
        "FRANCE": "FRANCJA",
        "SPAIN": "HISZPANIA",
        "ITALY": "WŁOCHY",
        "USA": "USA",
        "UK": "WIELKA BRYTANIA",
        "RUSSIA": "ROSJA",
        "JAPAN": "JAPONIA",
        "CHINA": "CHINY"
    }
    
    country_upper = country.upper()
    display_country = COUNTRY_TRANSLATIONS.get(country_upper, country_upper)

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)
    
    ax.text(0.5, 0.10, display_country, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=THEME['text'], linewidth=1, zorder=11)

    # --- POIs ---
    if pois:
        print(f"Drawing {len(pois)} POIs...")
        for poi in pois:
            p_lat, p_lon = poi['lat'], poi['lon']
            # Only draw if within bounds
            if (bbox[2] <= p_lat <= bbox[3]) and (bbox[0] <= p_lon <= bbox[1]): # Note: bbox needed from ax.get_xlim/ylim or calculated
                # Convert lat/lon to ax coords is automatic if we plot on ax with raw coords? 
                # No, osmnx plots projected data usually? 
                # Wait, ox.graph_from_point returns a graph in some CRS.
                # ox.plot_graph by default projects it.
                # If we want to plot points, we should project them too if the graph is projected.
                # By default ox.project_graph projects to UTM.
                # But here we didn't project the graph explicitly in script, but ox.graph_from_point loads it unprojected (WGS84) unless simplified/projected?
                # Actually ox.plot_graph plots in lat-lon by default if G is not projected.
                # Let's check if we project G. We don't see any 'ox.project_graph' call in existing code.
                # So G is likely in WGS84 (lat/lon). 
                # So we can plot directly using lat/lon, but remember X=lon, Y=lat.
                
                # Draw marker
                icon_type = poi.get('icon', 'default')
                
                # Check if icon_type is a file path or 'stadium' (which maps to stadium.png)
                icon_path = None
                if os.path.exists(icon_type):
                    icon_path = icon_type
                elif icon_type == 'stadium' and os.path.exists('stadium.png'):
                    icon_path = 'stadium.png'
                
                if icon_path:
                    try:
                        # Load image
                        img = plt.imread(icon_path)
                        
                        # Recolor: Change non-transparent pixels to accent color
                        if img.shape[2] == 4: # RGBA
                            # Create mask for non-transparent pixels (Alpha > 0)
                            mask = img[:, :, 3] > 0
                            
                            # Get accent color
                            accent_color = THEME.get('accent', 'red')
                            rgb = mcolors.to_rgb(accent_color)
                            
                            # Apply color preserving alpha
                            img = img.copy()
                            img[mask, 0] = rgb[0]
                            img[mask, 1] = rgb[1]
                            img[mask, 2] = rgb[2]
                        
                        # Create OffsetImage with configurable zoom and better interpolation
                        # Default zoom adjusted based on typical icon size (~500-1000px)
                        zoom_factor = icon_size / max(img.shape[0], img.shape[1])
                        imagebox = OffsetImage(img, zoom=zoom_factor, interpolation='lanczos')
                        
                        # Place image
                        ab = AnnotationBbox(imagebox, (p_lon, p_lat),
                                            frameon=False, pad=0, zorder=21)
                        ax.add_artist(ab)
                        
                    except Exception as e:
                        print(f"⚠ Failed to load/process icon {icon_path}: {e}")
                        # Fallback to marker
                        ax.plot(p_lon, p_lat, marker='o', markersize=15, color=THEME.get('accent', 'red'), 
                                markeredgecolor='white', markeredgewidth=2, zorder=20)
                else:
                    # Default marker or fallback
                    ax.plot(p_lon, p_lat, marker='o', markersize=10, color=THEME.get('accent', 'red'), 
                            markeredgecolor='white', markeredgewidth=1, zorder=20)
                
                # Draw Label
                # Offset label - move it down below the icon? Or keep top?
                # Stadium icon might be large, so let's move text a bit lower or higher?
                # Default behavior was top. Let's keep it but maybe increase offset if it clashes.
                # Actually, let's put it BELOW the icon for stadium
                va = 'top'
                offset_y = 0
                if icon_type == 'stadium':
                    va = 'top' # Text below point
                    # The text is drawn at (p_lon, p_lat). If image is centered there, text overlaps.
                    # We can't easily offset text in data coords without knowing scale.
                    # But we can assume a small offset.
                    # For now keep as is, it might overlap bottom of icon.
                    pass

                ax.text(p_lon, p_lat, f"\n{poi['label']}", 
                        color=THEME['text'], ha='center', va=va, 
                        fontproperties=font_coords, zorder=25,
                        path_effects=[pe.withStroke(linewidth=2, foreground=THEME['bg'])])

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_attr = FontProperties(family='monospace', size=8)
    
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME['text'], alpha=0.5, ha='right', va='bottom', 
            fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05,)

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --theme, -t       Theme name (default: feature_based)
  --distance, -d    Map radius in meters (default: 29000)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--poi', action='append', help='Custom POI: lat,lon,label,[icon_type] (can be used multiple times)')
    parser.add_argument('--lat-offset', type=float, default=0.0, help='Shift map center North/South (degrees). Positive = North.')
    parser.add_argument('--lon-offset', type=float, default=0.0, help='Shift map center East/West (degrees). Positive = East.')
    parser.add_argument('--draw-boundary', action='store_true', help='Draw city administrative boundary line')
    parser.add_argument('--icon-size', type=int, default=50, help='Target size of POI icons in pixels (default: 50)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],help='Output format for the poster (default: png)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(os.sys.argv) == 1:
        print_examples()
        os.sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        os.sys.exit(0)
    
    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        os.sys.exit(1)
    
    # Validated theme exists
    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        os.sys.exit(1)
    
    # Parse POIs
    pois = []
    if args.poi:
        for poi_str in args.poi:
            try:
                parts = poi_str.split(',')
                if len(parts) >= 3:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    label = parts[2]
                    icon = parts[3] if len(parts) > 3 else 'default'
                    pois.append({'lat': lat, 'lon': lon, 'label': label, 'icon': icon})
                else:
                    print(f"⚠ Invalid POI format: {poi_str}. Expected: lat,lon,label or lat,lon,label,icon")
            except ValueError:
                print(f"⚠ Invalid coordinates in POI: {poi_str}")

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)
    
    # Load theme
    THEME = load_theme(args.theme)
    
    # Get coordinates and generate poster
    try:
        base_coords = get_coordinates(args.city, args.country)
        # Apply offsets
        coords = (base_coords[0] + args.lat_offset, base_coords[1] + args.lon_offset)
        
        if args.lat_offset != 0 or args.lon_offset != 0:
            print(f"Applying offset: Lat {args.lat_offset:+.4f}, Lon {args.lon_offset:+.4f}")
            print(f"New center: {coords[0]:.6f}, {coords[1]:.6f}")

        output_file = generate_output_filename(args.city, args.theme, args.format)
        create_poster(args.city, args.country, coords, args.distance, output_file, output_format=args.format, pois=pois, draw_boundary=args.draw_boundary, icon_size=args.icon_size)
        
        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        os.sys.exit(1)