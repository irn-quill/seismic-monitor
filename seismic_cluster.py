"""
Seismic Cluster Detection — Bidirectional n+1 Algorithm
Networks: USGS + EMSC + ISC + GFZ Potsdam + IRIS/EarthScope
Outputs:
  - seismic_clusters.kml  (Google Earth Pro)
  - seismic_map.html      (Live Leaflet web map — open in any browser)
"""

import pandas as pd
import math
import time
import requests
import io
import json
import os
from datetime import datetime, timedelta, timezone

# ── Configuration ─────────────────────────────────────────────────────────────
DISTANCE_THRESHOLD_KM = 100
TIME_THRESHOLD_HRS    = 1.0
REFRESH_SECONDS       = 60
KML_OUTPUT_FILE       = "seismic_clusters.kml"
HTML_OUTPUT_FILE      = "seismic_map.html"
FAULT_CACHE_FILE      = "gem_global_faults.geojson"
LOOKBACK_HOURS        = 24

NETWORK_COLOURS = {
    'USGS':  '#ff3333',
    'EMSC':  '#33cc33',
    'ISC':   '#ff8800',
    'GFZ':   '#cc33ff',
    'IRIS':  '#00ccff',
    'MULTI': '#ffff00',
}
ISOLATED_COLOUR = '#aaddff'
# ──────────────────────────────────────────────────────────────────────────────


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ── Fault line loader ─────────────────────────────────────────────────────────

FAULT_SEGMENTS = []

def load_fault_lines():
    global FAULT_SEGMENTS
    url = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"

    if os.path.exists(FAULT_CACHE_FILE):
        print(f"  Faults : loaded from cache ({FAULT_CACHE_FILE})")
        with open(FAULT_CACHE_FILE, 'r') as f:
            data = json.load(f)
    else:
        print("  Faults : downloading USGS Quaternary Fault Database (~8MB)...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            with open(FAULT_CACHE_FILE, 'w') as f:
                json.dump(data, f)
            print(f"  Faults : saved to {FAULT_CACHE_FILE}")
        except Exception as e:
            print(f"  Faults : download failed ({e}) — proximity unavailable")
            return

    segments = []
    for feature in data.get('features', []):
        geom = feature.get('geometry', {})
        name = feature.get('properties', {}).get('name', feature.get('properties', {}).get('fault_name', 'Unknown'))
        coords = []
        if geom.get('type') == 'MultiLineString':
            for line in geom.get('coordinates', []):
                coords.append(line)
        elif geom.get('type') == 'LineString':
            coords.append(geom.get('coordinates', []))
        for line in coords:
            for k in range(len(line) - 1):
                try:
                    lon1, lat1 = float(line[k][0]),   float(line[k][1])
                    lon2, lat2 = float(line[k+1][0]), float(line[k+1][1])
                    segments.append((lat1, lon1, lat2, lon2, name))
                except (ValueError, IndexError, TypeError):
                    continue

    FAULT_SEGMENTS = segments
    print(f"  Faults : {len(FAULT_SEGMENTS):,} segments indexed")
    build_fault_grid()


def point_to_segment_km(plat, plon, lat1, lon1, lat2, lon2):
    d_a  = haversine_km(plat, plon, lat1, lon1)
    d_ab = haversine_km(lat1, lon1, lat2, lon2)
    if d_ab < 0.001:
        return d_a
    t = ((plat - lat1) * (lat2 - lat1) + (plon - lon1) * (lon2 - lon1)) / \
        ((lat2 - lat1)**2 + (lon2 - lon1)**2 + 1e-12)
    t = max(0.0, min(1.0, t))
    proj_lat = lat1 + t * (lat2 - lat1)
    proj_lon = lon1 + t * (lon2 - lon1)
    return haversine_km(plat, plon, proj_lat, proj_lon)


# Spatial grid index for fast fault lookup — built once after load_fault_lines()
FAULT_GRID = {}   # (int(lat), int(lon)) -> list of segment indices
FAULT_GRID_BUILT = False

def build_fault_grid():
    global FAULT_GRID, FAULT_GRID_BUILT
    FAULT_GRID = {}
    for idx, seg in enumerate(FAULT_SEGMENTS):
        slat1, slon1, slat2, slon2, _ = seg
        mid_lat = int((slat1 + slat2) / 2)
        mid_lon = int((slon1 + slon2) / 2)
        for dlat in range(-2, 3):
            for dlon in range(-2, 3):
                key = (mid_lat + dlat, mid_lon + dlon)
                if key not in FAULT_GRID:
                    FAULT_GRID[key] = []
                FAULT_GRID[key].append(idx)
    FAULT_GRID_BUILT = True
    print(f"  Faults : spatial grid built ({len(FAULT_GRID)} cells)")


def nearest_fault_km(lat, lon):
    if not FAULT_SEGMENTS:
        return None, None
    if not FAULT_GRID_BUILT:
        build_fault_grid()
    key = (int(lat), int(lon))
    candidates = set()
    for dlat in range(-2, 3):
        for dlon in range(-2, 3):
            candidates.update(FAULT_GRID.get((key[0]+dlat, key[1]+dlon), []))
    if not candidates:
        return None, None
    best_dist = float('inf')
    best_name = None
    for idx in candidates:
        slat1, slon1, slat2, slon2, sname = FAULT_SEGMENTS[idx]
        d = point_to_segment_km(lat, lon, slat1, slon1, slat2, slon2)
        if d < best_dist:
            best_dist = d
            best_name = sname
    return (round(best_dist, 1), best_name) if best_dist < float('inf') else (None, None)


def fault_proximity_label(dist_km):
    if dist_km is None:  return 'unknown'
    if dist_km < 10:     return 'ON FAULT'
    if dist_km < 50:     return 'near fault'
    if dist_km < 150:    return 'fault zone'
    return 'inter-fault'


def fault_str_for(row):
    fd = row.get('fault_dist_km')
    if fd is not None and pd.notna(fd):
        return f"{fd:.0f}km from {row.get('fault_name', '')} ({row.get('fault_proximity', '')})"
    return 'unavailable'


# ── Network fetchers ──────────────────────────────────────────────────────────

def fetch_usgs():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df['network'] = 'USGS'
    return df[['time', 'latitude', 'longitude', 'mag', 'depth', 'place', 'network']]


def _fetch_fdsn(base_url, network_name):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    url = (
        f"{base_url}query"
        f"?starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
        "&minmag=1&format=text"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [l for l in r.text.strip().split('\n') if not l.startswith('#') and l.strip()]
    if not lines:
        return pd.DataFrame(columns=['time','latitude','longitude','mag','depth','place','network'])
    rows = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 11:
            try:
                rows.append({
                    'time':      parts[1].strip(),
                    'latitude':  float(parts[2]),
                    'longitude': float(parts[3]),
                    'depth':     float(parts[4]) if parts[4].strip() else None,
                    'mag':       float(parts[10]) if parts[10].strip() else None,
                    'place':     parts[12].strip() if len(parts) > 12 else '',
                    'network':   network_name
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def fetch_emsc():
    return _fetch_fdsn("https://www.seismicportal.eu/fdsnws/event/1/", "EMSC")

def fetch_isc():
    return _fetch_fdsn("http://www.isc.ac.uk/fdsnws/event/1/", "ISC")

def fetch_gfz():
    return _fetch_fdsn("http://geofon.gfz-potsdam.de/fdsnws/event/1/", "GFZ")

def fetch_iris():
    return _fetch_fdsn("https://service.iris.edu/fdsnws/event/1/", "IRIS")


def fetch_all_networks():
    frames = []
    for name, fn in [('USGS', fetch_usgs), ('EMSC', fetch_emsc),
                     ('ISC',  fetch_isc),  ('GFZ',  fetch_gfz),
                     ('IRIS', fetch_iris)]:
        try:
            df = fn()
            print(f"  {name:4} : {len(df)} events")
            frames.append(df)
        except Exception as e:
            print(f"  {name:4} : failed ({e})")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['latitude', 'longitude', 'mag', 'time'])
    combined['parsed_time'] = pd.to_datetime(combined['time'], utc=True, errors='coerce')
    combined = combined.dropna(subset=['parsed_time'])
    combined = combined.sort_values('parsed_time').reset_index(drop=True)

    keep          = [True] * len(combined)
    network_label = list(combined['network'])

    for i in range(len(combined)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(combined)):
            if not keep[j]:
                continue
            dt = abs((combined.at[j, 'parsed_time'] - combined.at[i, 'parsed_time']).total_seconds() / 60)
            if dt > 5:
                break
            dist = haversine_km(combined.at[i, 'latitude'], combined.at[i, 'longitude'],
                                 combined.at[j, 'latitude'], combined.at[j, 'longitude'])
            if dist < 10:
                if combined.at[j, 'network'] != combined.at[i, 'network']:
                    network_label[i] = 'MULTI'
                keep[j] = False

    combined = combined[keep].copy().reset_index(drop=True)
    combined['network'] = network_label[:len(combined)]
    print(f"  Combined (deduplicated): {len(combined)} unique events")
    return combined


# ── Algorithm ─────────────────────────────────────────────────────────────────

def run_algorithm(df):
    df = df.copy().sort_values('parsed_time').reset_index(drop=True)
    ground = df['parsed_time'].min()
    df['ground_offset_hrs'] = (df['parsed_time'] - ground).dt.total_seconds() / 3600

    n = len(df)
    dist_prev = [None]*n; dist_next = [None]*n
    time_prev = [None]*n; time_next = [None]*n

    for i in range(n):
        lat, lon, t = df.at[i,'latitude'], df.at[i,'longitude'], df.at[i,'parsed_time']
        if i > 0:
            dist_prev[i] = haversine_km(lat, lon, df.at[i-1,'latitude'], df.at[i-1,'longitude'])
            time_prev[i] = abs((t - df.at[i-1,'parsed_time']).total_seconds() / 3600)
        if i < n-1:
            dist_next[i] = haversine_km(lat, lon, df.at[i+1,'latitude'], df.at[i+1,'longitude'])
            time_next[i] = abs((df.at[i+1,'parsed_time'] - t).total_seconds() / 3600)

    df['dist_prev_km'] = dist_prev
    df['dist_next_km'] = dist_next
    df['time_prev_hrs'] = time_prev
    df['time_next_hrs'] = time_next

    flags = []
    for i in range(n):
        prev = dist_prev[i] is not None and dist_prev[i] < DISTANCE_THRESHOLD_KM and time_prev[i] < TIME_THRESHOLD_HRS
        nxt  = dist_next[i] is not None and dist_next[i] < DISTANCE_THRESHOLD_KM and time_next[i] < TIME_THRESHOLD_HRS
        flags.append('CLUSTER' if (prev or nxt) else '—')
    df['cluster_flag'] = flags

    # ── Depth analysis ────────────────────────────────────────────────────────
    def depth_class(d):
        if pd.isna(d):  return 'unknown'
        if d < 20:      return 'very shallow'
        if d < 70:      return 'shallow'
        if d < 300:     return 'intermediate'
        return 'deep'

    df['depth_class'] = df['depth'].apply(depth_class)

    depth_delta_prev = [None]*n
    depth_delta_next = [None]*n
    for i in range(n):
        if df.at[i,'cluster_flag'] != 'CLUSTER':
            continue
        d = df.at[i,'depth']
        if pd.isna(d):
            continue
        if i > 0 and pd.notna(df.at[i-1,'depth']):
            depth_delta_prev[i] = d - df.at[i-1,'depth']
        if i < n-1 and pd.notna(df.at[i+1,'depth']):
            depth_delta_next[i] = df.at[i+1,'depth'] - d

    df['depth_delta_prev_km'] = depth_delta_prev
    df['depth_delta_next_km'] = depth_delta_next

    def precursor_flag(row):
        if row['cluster_flag'] != 'CLUSTER': return '—'
        dp, dn = row['depth_delta_prev_km'], row['depth_delta_next_km']
        if pd.notna(dp) and pd.notna(dn) and dp < -2 and dn < -2: return 'SHALLOWING'
        if pd.notna(dp) and dp < -5: return 'SHALLOWING'
        return '—'

    df['depth_precursor'] = df.apply(precursor_flag, axis=1)

    # ── Fault proximity ───────────────────────────────────────────────────────
    if FAULT_SEGMENTS:
        fault_dists  = []
        fault_names  = []
        fault_labels = []
        for _, row in df.iterrows():
            d, name = nearest_fault_km(row['latitude'], row['longitude'])
            fault_dists.append(d)
            fault_names.append(name)
            fault_labels.append(fault_proximity_label(d))
        df['fault_dist_km']   = fault_dists
        df['fault_name']      = fault_names
        df['fault_proximity'] = fault_labels
    else:
        df['fault_dist_km']   = None
        df['fault_name']      = None
        df['fault_proximity'] = 'unavailable'

    return df




# ── Atmospheric pressure correlation ──────────────────────────────────────────
# Uses Open-Meteo — free, no API key, global coverage
# For each cluster event, fetches surface pressure for the 48h before the event
# Flags if pressure dropped significantly in the window before the event
# Note: correlation between pressure and seismicity is contested but published

PRESSURE_CACHE = {}   # (round_lat, round_lon, date) -> pressure series

def fetch_pressure_for_event(lat, lon, event_time):
    """Get hourly surface pressure for 48h before event_time at lat/lon."""
    # Round to 1dp to allow cache reuse for nearby events
    rlat = round(lat, 1)
    rlon = round(lon, 1)
    date_end   = event_time.strftime('%Y-%m-%d')
    date_start = (event_time - timedelta(hours=48)).strftime('%Y-%m-%d')
    cache_key  = (rlat, rlon, date_start, date_end)

    if cache_key in PRESSURE_CACHE:
        return PRESSURE_CACHE[cache_key]

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={rlat}&longitude={rlon}"
            f"&hourly=surface_pressure"
            f"&start_date={date_start}&end_date={date_end}"
            f"&timezone=UTC"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        times     = data['hourly']['time']
        pressures = data['hourly']['surface_pressure']
        series = list(zip(times, pressures))
        PRESSURE_CACHE[cache_key] = series
        return series
    except Exception:
        PRESSURE_CACHE[cache_key] = None
        return None


def pressure_delta_hpa(series, event_time):
    """
    Calculate pressure change in the 24h window before the event.
    Returns (pressure_at_event, change_24h, flag)
    Negative change = pressure dropped before event.
    """
    if not series:
        return None, None, '—'

    event_str = event_time.strftime('%Y-%m-%dT%H:00')
    minus24_str = (event_time - timedelta(hours=24)).strftime('%Y-%m-%dT%H:00')

    p_event  = None
    p_minus24 = None

    for t, p in series:
        if t == event_str:
            p_event = p
        if t == minus24_str:
            p_minus24 = p

    if p_event is None or p_minus24 is None:
        return p_event, None, '—'

    delta = p_event - p_minus24

    # Flag significant drops — threshold based on published research
    if delta < -8:
        flag = '🌀 PRESSURE DROP'
    elif delta < -4:
        flag = 'pressure falling'
    elif delta > 4:
        flag = 'pressure rising'
    else:
        flag = '—'

    return round(p_event, 1), round(delta, 1), flag


def add_pressure_correlation(df):
    """Add pressure data to cluster events only (to limit API calls)."""
    pressure_hpa   = [None] * len(df)
    pressure_delta = [None] * len(df)
    pressure_flag  = ['—'] * len(df)

    cluster_indices = df.index[df['cluster_flag'] == 'CLUSTER'].tolist()
    if not cluster_indices:
        df['pressure_hpa']   = pressure_hpa
        df['pressure_delta'] = pressure_delta
        df['pressure_flag']  = pressure_flag
        return df

    print(f"  Pressure: fetching for {len(cluster_indices)} cluster events...")
    for i in cluster_indices:
        row = df.loc[i]
        series = fetch_pressure_for_event(row['latitude'], row['longitude'], row['parsed_time'])
        p, delta, flag = pressure_delta_hpa(series, row['parsed_time'])
        pressure_hpa[i]   = p
        pressure_delta[i] = delta
        pressure_flag[i]  = flag

    df['pressure_hpa']   = pressure_hpa
    df['pressure_delta'] = pressure_delta
    df['pressure_flag']  = pressure_flag
    return df
# ─────────────────────────────────────────────────────────────────────────────

# ── HTML Leaflet map ──────────────────────────────────────────────────────────

def write_html(df):
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    total     = len(df)
    clusters  = int((df['cluster_flag'] == 'CLUSTER').sum())
    multi     = int(((df['cluster_flag'] == 'CLUSTER') & (df['network'] == 'MULTI')).sum())
    on_fault  = int((df['fault_proximity'] == 'ON FAULT').sum()) if 'fault_proximity' in df.columns else 0

    markers = []
    for _, row in df.iterrows():
        is_cluster   = row['cluster_flag'] == 'CLUSTER'
        is_precursor = row.get('depth_precursor') == 'SHALLOWING'
        net          = row.get('network', 'USGS')
        colour       = NETWORK_COLOURS.get(net, '#ff3333') if is_cluster else ISOLATED_COLOUR
        radius       = max(4, float(row['mag']) * 3) if is_cluster else max(3, float(row['mag']) * 2)
        depth        = row.get('depth', None)
        depth_str    = f"{depth:.0f}km ({row.get('depth_class','?')})" if pd.notna(depth) else '?'
        dp           = f"{row['dist_prev_km']:.0f}km" if pd.notna(row.get('dist_prev_km')) else '?'
        dn           = f"{row['dist_next_km']:.0f}km" if pd.notna(row.get('dist_next_km')) else '?'
        place        = str(row.get('place', '')).replace("'", "\\'")
        t            = str(row['parsed_time'])[:19]
        warning      = ' ⚠️ SHALLOWING' if is_precursor else ''
        popup = (f"<b>M{row['mag']} — {place}</b><br>"
                 f"Network: {net}<br>"
                 f"Time: {t} UTC<br>"
                 f"Depth: {depth_str}{warning}<br>"
                 f"Prev: {dp} | Next: {dn}<br>"
                 f"Fault: {fault_str_for(row)}<br>"
                 f"Pressure: {row.get('pressure_hpa','?') or '?'} hPa (24h change: {row.get('pressure_delta','?') or '?'} hPa) {row.get('pressure_flag','') or ''}<br>"
                 f"<b>{'🔴 CLUSTER' if is_cluster else '⚪ Isolated'}</b>")
        markers.append({
            'lat':     float(row['latitude']),
            'lon':     float(row['longitude']),
            'colour':  colour,
            'radius':  radius,
            'popup':   popup,
            'cluster': is_cluster,
        })

    markers_json = json.dumps(markers)

    legend_items = ''.join([
        f'<div><span style="background:{c};display:inline-block;width:12px;height:12px;'
        f'border-radius:50%;margin-right:6px;"></span>{n}</div>'
        for n, c in NETWORK_COLOURS.items()
    ])
    legend_items += (f'<div><span style="background:{ISOLATED_COLOUR};display:inline-block;'
                     f'width:12px;height:12px;border-radius:50%;margin-right:6px;"></span>Isolated</div>')

    on_fault_line = f'<div class="on-fault">📍 {on_fault} ON FAULT</div>' if on_fault > 0 else ''
    multi_line    = f'<div class="multi">⭐ {multi} multi-network</div>' if multi > 0 else ''

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Global Seismic Cluster Monitor</title>
  <meta http-equiv="refresh" content="60">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0a0a0f; color:#eee; font-family: 'Helvetica Neue', sans-serif; }}
    #map {{ width:100vw; height:100vh; }}
    #panel {{
      position:absolute; top:16px; left:16px; z-index:1000;
      background:rgba(10,10,20,0.85); border:1px solid #333;
      border-radius:10px; padding:16px; min-width:220px;
      backdrop-filter: blur(8px);
    }}
    #panel h2       {{ font-size:14px; color:#fff; margin-bottom:10px; letter-spacing:1px; }}
    #panel .stat    {{ font-size:22px; font-weight:bold; color:#ff4444; }}
    #panel .label   {{ font-size:11px; color:#999; margin-bottom:8px; }}
    #panel .multi   {{ color:#ffff00; font-size:13px; font-weight:bold; }}
    #panel .on-fault{{ color:#ff8800; font-size:13px; font-weight:bold; }}
    #legend {{ margin-top:12px; border-top:1px solid #333; padding-top:10px; font-size:12px; line-height:22px; }}
    #timestamp {{ font-size:10px; color:#666; margin-top:10px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h2>⚡ SEISMIC CLUSTER MONITOR</h2>
  <div class="stat">{clusters}</div>
  <div class="label">active clusters / {total} global events (24h)</div>
  {multi_line}
  {on_fault_line}
  <div id="legend">{legend_items}</div>
  <div id="timestamp">Updated: {timestamp}<br>Auto-refreshes every 60s</div>
</div>
<script>
  var map = L.map('map', {{ center: [20, 0], zoom: 2, zoomControl: true }});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
  }}).addTo(map);

  var markers = {markers_json};
  markers.forEach(function(m) {{
    L.circleMarker([m.lat, m.lon], {{
      radius:      m.radius,
      fillColor:   m.colour,
      color:       m.cluster ? '#fff' : 'transparent',
      weight:      m.cluster ? 1 : 0,
      opacity:     0.9,
      fillOpacity: m.cluster ? 0.85 : 0.5
    }}).addTo(map).bindPopup(m.popup);
  }});
</script>
</body>
</html>"""

    with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  → Map saved: {HTML_OUTPUT_FILE}")


# ── KML ───────────────────────────────────────────────────────────────────────

def write_kml(df):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '<Document>', f'  <n>Seismic Clusters — {ts}</n>']

    for net, colour in NETWORK_COLOURS.items():
        h = colour.lstrip('#')
        kml_colour = f'ff{h[4:6]}{h[2:4]}{h[0:2]}'
        lines.append(f'''  <Style id="cluster_{net}">
    <IconStyle><color>{kml_colour}</color><scale>1.2</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
    </IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>''')

    lines.append('''  <Style id="isolated">
    <IconStyle><color>ffddaa88</color><scale>0.7</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href></Icon>
    </IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>''')

    lines.append('  <Folder><n>🔴 Clusters</n>')
    for _, row in df[df['cluster_flag'] == 'CLUSTER'].iterrows():
        net   = row.get('network', 'USGS')
        style = f"cluster_{net}" if net in NETWORK_COLOURS else "cluster_USGS"
        place = str(row.get('place', 'Unknown'))
        dp    = f"{row['dist_prev_km']:.1f}" if pd.notna(row.get('dist_prev_km')) else '?'
        dn    = f"{row['dist_next_km']:.1f}" if pd.notna(row.get('dist_next_km')) else '?'
        pre   = ' ⚠️ SHALLOWING' if row.get('depth_precursor') == 'SHALLOWING' else ''
        lines.append(f'''    <Placemark>
      <n>M{row['mag']} — {place}</n>
      <description>Network: {net}
Time: {str(row['parsed_time'])[:19]} UTC
Magnitude: {row['mag']}  Depth: {row.get('depth','?')} km ({row.get('depth_class','?')}){pre}
Prev: {dp} km | Next: {dn} km
Fault: {fault_str_for(row)}</description>
      <styleUrl>#{style}</styleUrl>
      <Point><coordinates>{row['longitude']},{row['latitude']},0</coordinates></Point>
    </Placemark>''')
    lines.append('  </Folder>')

    lines.append('  <Folder><n>⚪ Isolated</n>')
    for _, row in df[df['cluster_flag'] == '—'].iterrows():
        lines.append(f'''    <Placemark>
      <n>M{row['mag']} — {str(row.get('place',''))}</n>
      <description>Network: {row.get('network','?')}
Time: {str(row['parsed_time'])[:19]} UTC
Magnitude: {row['mag']}  Depth: {row.get('depth','?')} km
Fault: {fault_str_for(row)}</description>
      <styleUrl>#isolated</styleUrl>
      <Point><coordinates>{row['longitude']},{row['latitude']},0</coordinates></Point>
    </Placemark>''')
    lines.append('  </Folder>')
    lines += ['</Document>', '</kml>']

    with open(KML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  → KML saved: {KML_OUTPUT_FILE}")


# ── Report ────────────────────────────────────────────────────────────────────

def report(df):
    total     = len(df)
    clusters  = (df['cluster_flag'] == 'CLUSTER').sum()
    multi     = ((df['cluster_flag'] == 'CLUSTER') & (df['network'] == 'MULTI')).sum()
    precursor = (df['depth_precursor'] == 'SHALLOWING').sum() if 'depth_precursor' in df.columns else 0
    shallow   = df['depth_class'].isin(['very shallow','shallow']).sum() if 'depth_class' in df.columns else 0
    on_fault  = (df['fault_proximity'] == 'ON FAULT').sum() if 'fault_proximity' in df.columns else 0

    print()
    print("═" * 55)
    print(f"  Ground timestamp : {df['parsed_time'].min().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Latest event     : {df['parsed_time'].max().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Total events     : {total}")
    print(f"  Clustered        : {clusters}  ({100*clusters/total:.1f}%)")
    print(f"  Shallow events   : {shallow} (depth < 70km)")
    if on_fault > 0:
        print(f"  📍 On fault      : {on_fault}  ← sitting directly on a known fault")
    if multi > 0:
        print(f"  ⭐ Multi-network : {multi}  ← confirmed by 2+ independent networks")
    pressure_drops = (df['pressure_flag'] == '🌀 PRESSURE DROP').sum() if 'pressure_flag' in df.columns else 0
    if precursor > 0:
        print(f"  ⚠️  Shallowing    : {precursor}  ← depth migration detected")
    if pressure_drops > 0:
        print(f"  🌀 Pressure drop : {pressure_drops}  ← significant pressure fall before event")
    print()

    if clusters > 0:
        print("  Most recent cluster events:")
        for _, row in df[df['cluster_flag'] == 'CLUSTER'].tail(10).iterrows():
            place = str(row.get('place', 'Unknown'))[:35]
            dp    = f"{row['dist_prev_km']:.0f}" if pd.notna(row.get('dist_prev_km')) else '?'
            dn    = f"{row['dist_next_km']:.0f}" if pd.notna(row.get('dist_next_km')) else '?'
            dep   = f"{row['depth']:.0f}km" if pd.notna(row.get('depth')) else '?'
            dc    = row.get('depth_class', '?')
            pre   = ' ⚠️' if row.get('depth_precursor') == 'SHALLOWING' else ''
            fp    = row.get('fault_proximity', '')
            fd    = f" [{fp}]" if fp and fp not in ('unavailable', 'unknown') else ''
            pf = f" {row['pressure_flag']}" if row.get('pressure_flag') and row.get('pressure_flag') not in ('—', None) else ""
            print(f"    [{row.get('network','?'):5}] M{row['mag']:.1f}  {place}  "
                  f"[prev:{dp}km|next:{dn}km] {dep} ({dc}){pre}{fd}{pf}")
    print("═" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

"""
Seismic Cluster Detection — Bidirectional n+1 Algorithm
Networks: USGS + EMSC + ISC + GFZ Potsdam + IRIS/EarthScope
Outputs:
  - seismic_clusters.kml  (Google Earth Pro)
  - seismic_map.html      (Live Leaflet web map — open in any browser)
"""

import pandas as pd
import math
import time
import requests
import io
import json
import os
from datetime import datetime, timedelta, timezone

# ── Configuration ─────────────────────────────────────────────────────────────
DISTANCE_THRESHOLD_KM = 100
TIME_THRESHOLD_HRS    = 1.0
REFRESH_SECONDS       = 60
KML_OUTPUT_FILE       = "seismic_clusters.kml"
HTML_OUTPUT_FILE      = "seismic_map.html"
FAULT_CACHE_FILE      = "gem_global_faults.geojson"
LOOKBACK_HOURS        = 24

NETWORK_COLOURS = {
    'USGS':  '#ff3333',
    'EMSC':  '#33cc33',
    'ISC':   '#ff8800',
    'GFZ':   '#cc33ff',
    'IRIS':  '#00ccff',
    'MULTI': '#ffff00',
}
ISOLATED_COLOUR = '#aaddff'
# ──────────────────────────────────────────────────────────────────────────────


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ── Fault line loader ─────────────────────────────────────────────────────────

FAULT_SEGMENTS = []

def load_fault_lines():
    global FAULT_SEGMENTS
    url = "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults.geojson"

    if os.path.exists(FAULT_CACHE_FILE):
        print(f"  Faults : loaded from cache ({FAULT_CACHE_FILE})")
        with open(FAULT_CACHE_FILE, 'r') as f:
            data = json.load(f)
    else:
        print("  Faults : downloading USGS Quaternary Fault Database (~8MB)...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            with open(FAULT_CACHE_FILE, 'w') as f:
                json.dump(data, f)
            print(f"  Faults : saved to {FAULT_CACHE_FILE}")
        except Exception as e:
            print(f"  Faults : download failed ({e}) — proximity unavailable")
            return

    segments = []
    for feature in data.get('features', []):
        geom = feature.get('geometry', {})
        name = feature.get('properties', {}).get('name', feature.get('properties', {}).get('fault_name', 'Unknown'))
        coords = []
        if geom.get('type') == 'MultiLineString':
            for line in geom.get('coordinates', []):
                coords.append(line)
        elif geom.get('type') == 'LineString':
            coords.append(geom.get('coordinates', []))
        for line in coords:
            for k in range(len(line) - 1):
                try:
                    lon1, lat1 = float(line[k][0]),   float(line[k][1])
                    lon2, lat2 = float(line[k+1][0]), float(line[k+1][1])
                    segments.append((lat1, lon1, lat2, lon2, name))
                except (ValueError, IndexError, TypeError):
                    continue

    FAULT_SEGMENTS = segments
    print(f"  Faults : {len(FAULT_SEGMENTS):,} segments indexed")
    build_fault_grid()


def point_to_segment_km(plat, plon, lat1, lon1, lat2, lon2):
    d_a  = haversine_km(plat, plon, lat1, lon1)
    d_ab = haversine_km(lat1, lon1, lat2, lon2)
    if d_ab < 0.001:
        return d_a
    t = ((plat - lat1) * (lat2 - lat1) + (plon - lon1) * (lon2 - lon1)) / \
        ((lat2 - lat1)**2 + (lon2 - lon1)**2 + 1e-12)
    t = max(0.0, min(1.0, t))
    proj_lat = lat1 + t * (lat2 - lat1)
    proj_lon = lon1 + t * (lon2 - lon1)
    return haversine_km(plat, plon, proj_lat, proj_lon)


# Spatial grid index for fast fault lookup — built once after load_fault_lines()
FAULT_GRID = {}   # (int(lat), int(lon)) -> list of segment indices
FAULT_GRID_BUILT = False

def build_fault_grid():
    global FAULT_GRID, FAULT_GRID_BUILT
    FAULT_GRID = {}
    for idx, seg in enumerate(FAULT_SEGMENTS):
        slat1, slon1, slat2, slon2, _ = seg
        mid_lat = int((slat1 + slat2) / 2)
        mid_lon = int((slon1 + slon2) / 2)
        for dlat in range(-2, 3):
            for dlon in range(-2, 3):
                key = (mid_lat + dlat, mid_lon + dlon)
                if key not in FAULT_GRID:
                    FAULT_GRID[key] = []
                FAULT_GRID[key].append(idx)
    FAULT_GRID_BUILT = True
    print(f"  Faults : spatial grid built ({len(FAULT_GRID)} cells)")


def nearest_fault_km(lat, lon):
    if not FAULT_SEGMENTS:
        return None, None
    if not FAULT_GRID_BUILT:
        build_fault_grid()
    key = (int(lat), int(lon))
    candidates = set()
    for dlat in range(-2, 3):
        for dlon in range(-2, 3):
            candidates.update(FAULT_GRID.get((key[0]+dlat, key[1]+dlon), []))
    if not candidates:
        return None, None
    best_dist = float('inf')
    best_name = None
    for idx in candidates:
        slat1, slon1, slat2, slon2, sname = FAULT_SEGMENTS[idx]
        d = point_to_segment_km(lat, lon, slat1, slon1, slat2, slon2)
        if d < best_dist:
            best_dist = d
            best_name = sname
    return (round(best_dist, 1), best_name) if best_dist < float('inf') else (None, None)


def fault_proximity_label(dist_km):
    if dist_km is None:  return 'unknown'
    if dist_km < 10:     return 'ON FAULT'
    if dist_km < 50:     return 'near fault'
    if dist_km < 150:    return 'fault zone'
    return 'inter-fault'


def fault_str_for(row):
    fd = row.get('fault_dist_km')
    if fd is not None and pd.notna(fd):
        return f"{fd:.0f}km from {row.get('fault_name', '')} ({row.get('fault_proximity', '')})"
    return 'unavailable'


# ── Network fetchers ──────────────────────────────────────────────────────────

def fetch_usgs():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df['network'] = 'USGS'
    return df[['time', 'latitude', 'longitude', 'mag', 'depth', 'place', 'network']]


def _fetch_fdsn(base_url, network_name):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    url = (
        f"{base_url}query"
        f"?starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
        "&minmag=1&format=text"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = [l for l in r.text.strip().split('\n') if not l.startswith('#') and l.strip()]
    if not lines:
        return pd.DataFrame(columns=['time','latitude','longitude','mag','depth','place','network'])
    rows = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 11:
            try:
                rows.append({
                    'time':      parts[1].strip(),
                    'latitude':  float(parts[2]),
                    'longitude': float(parts[3]),
                    'depth':     float(parts[4]) if parts[4].strip() else None,
                    'mag':       float(parts[10]) if parts[10].strip() else None,
                    'place':     parts[12].strip() if len(parts) > 12 else '',
                    'network':   network_name
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def fetch_emsc():
    return _fetch_fdsn("https://www.seismicportal.eu/fdsnws/event/1/", "EMSC")

def fetch_isc():
    return _fetch_fdsn("http://www.isc.ac.uk/fdsnws/event/1/", "ISC")

def fetch_gfz():
    return _fetch_fdsn("http://geofon.gfz-potsdam.de/fdsnws/event/1/", "GFZ")

def fetch_iris():
    return _fetch_fdsn("https://service.iris.edu/fdsnws/event/1/", "IRIS")


def fetch_all_networks():
    frames = []
    for name, fn in [('USGS', fetch_usgs), ('EMSC', fetch_emsc),
                     ('ISC',  fetch_isc),  ('GFZ',  fetch_gfz),
                     ('IRIS', fetch_iris)]:
        try:
            df = fn()
            print(f"  {name:4} : {len(df)} events")
            frames.append(df)
        except Exception as e:
            print(f"  {name:4} : failed ({e})")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['latitude', 'longitude', 'mag', 'time'])
    combined['parsed_time'] = pd.to_datetime(combined['time'], utc=True, errors='coerce')
    combined = combined.dropna(subset=['parsed_time'])
    combined = combined.sort_values('parsed_time').reset_index(drop=True)

    keep          = [True] * len(combined)
    network_label = list(combined['network'])

    for i in range(len(combined)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(combined)):
            if not keep[j]:
                continue
            dt = abs((combined.at[j, 'parsed_time'] - combined.at[i, 'parsed_time']).total_seconds() / 60)
            if dt > 5:
                break
            dist = haversine_km(combined.at[i, 'latitude'], combined.at[i, 'longitude'],
                                 combined.at[j, 'latitude'], combined.at[j, 'longitude'])
            if dist < 10:
                if combined.at[j, 'network'] != combined.at[i, 'network']:
                    network_label[i] = 'MULTI'
                keep[j] = False

    combined = combined[keep].copy().reset_index(drop=True)
    combined['network'] = network_label[:len(combined)]
    print(f"  Combined (deduplicated): {len(combined)} unique events")
    return combined


# ── Algorithm ─────────────────────────────────────────────────────────────────

def run_algorithm(df):
    df = df.copy().sort_values('parsed_time').reset_index(drop=True)
    ground = df['parsed_time'].min()
    df['ground_offset_hrs'] = (df['parsed_time'] - ground).dt.total_seconds() / 3600

    n = len(df)
    dist_prev = [None]*n; dist_next = [None]*n
    time_prev = [None]*n; time_next = [None]*n

    for i in range(n):
        lat, lon, t = df.at[i,'latitude'], df.at[i,'longitude'], df.at[i,'parsed_time']
        if i > 0:
            dist_prev[i] = haversine_km(lat, lon, df.at[i-1,'latitude'], df.at[i-1,'longitude'])
            time_prev[i] = abs((t - df.at[i-1,'parsed_time']).total_seconds() / 3600)
        if i < n-1:
            dist_next[i] = haversine_km(lat, lon, df.at[i+1,'latitude'], df.at[i+1,'longitude'])
            time_next[i] = abs((df.at[i+1,'parsed_time'] - t).total_seconds() / 3600)

    df['dist_prev_km'] = dist_prev
    df['dist_next_km'] = dist_next
    df['time_prev_hrs'] = time_prev
    df['time_next_hrs'] = time_next

    flags = []
    for i in range(n):
        prev = dist_prev[i] is not None and dist_prev[i] < DISTANCE_THRESHOLD_KM and time_prev[i] < TIME_THRESHOLD_HRS
        nxt  = dist_next[i] is not None and dist_next[i] < DISTANCE_THRESHOLD_KM and time_next[i] < TIME_THRESHOLD_HRS
        flags.append('CLUSTER' if (prev or nxt) else '—')
    df['cluster_flag'] = flags

    # ── Depth analysis ────────────────────────────────────────────────────────
    def depth_class(d):
        if pd.isna(d):  return 'unknown'
        if d < 20:      return 'very shallow'
        if d < 70:      return 'shallow'
        if d < 300:     return 'intermediate'
        return 'deep'

    df['depth_class'] = df['depth'].apply(depth_class)

    depth_delta_prev = [None]*n
    depth_delta_next = [None]*n
    for i in range(n):
        if df.at[i,'cluster_flag'] != 'CLUSTER':
            continue
        d = df.at[i,'depth']
        if pd.isna(d):
            continue
        if i > 0 and pd.notna(df.at[i-1,'depth']):
            depth_delta_prev[i] = d - df.at[i-1,'depth']
        if i < n-1 and pd.notna(df.at[i+1,'depth']):
            depth_delta_next[i] = df.at[i+1,'depth'] - d

    df['depth_delta_prev_km'] = depth_delta_prev
    df['depth_delta_next_km'] = depth_delta_next

    def precursor_flag(row):
        if row['cluster_flag'] != 'CLUSTER': return '—'
        dp, dn = row['depth_delta_prev_km'], row['depth_delta_next_km']
        if pd.notna(dp) and pd.notna(dn) and dp < -2 and dn < -2: return 'SHALLOWING'
        if pd.notna(dp) and dp < -5: return 'SHALLOWING'
        return '—'

    df['depth_precursor'] = df.apply(precursor_flag, axis=1)

    # ── Fault proximity ───────────────────────────────────────────────────────
    if FAULT_SEGMENTS:
        fault_dists  = []
        fault_names  = []
        fault_labels = []
        for _, row in df.iterrows():
            d, name = nearest_fault_km(row['latitude'], row['longitude'])
            fault_dists.append(d)
            fault_names.append(name)
            fault_labels.append(fault_proximity_label(d))
        df['fault_dist_km']   = fault_dists
        df['fault_name']      = fault_names
        df['fault_proximity'] = fault_labels
    else:
        df['fault_dist_km']   = None
        df['fault_name']      = None
        df['fault_proximity'] = 'unavailable'

    return df




# ── Atmospheric pressure correlation ──────────────────────────────────────────
# Uses Open-Meteo — free, no API key, global coverage
# For each cluster event, fetches surface pressure for the 48h before the event
# Flags if pressure dropped significantly in the window before the event
# Note: correlation between pressure and seismicity is contested but published

PRESSURE_CACHE = {}   # (round_lat, round_lon, date) -> pressure series

def fetch_pressure_for_event(lat, lon, event_time):
    """Get hourly surface pressure for 48h before event_time at lat/lon."""
    # Round to 1dp to allow cache reuse for nearby events
    rlat = round(lat, 1)
    rlon = round(lon, 1)
    date_end   = event_time.strftime('%Y-%m-%d')
    date_start = (event_time - timedelta(hours=48)).strftime('%Y-%m-%d')
    cache_key  = (rlat, rlon, date_start, date_end)

    if cache_key in PRESSURE_CACHE:
        return PRESSURE_CACHE[cache_key]

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={rlat}&longitude={rlon}"
            f"&hourly=surface_pressure"
            f"&start_date={date_start}&end_date={date_end}"
            f"&timezone=UTC"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        times     = data['hourly']['time']
        pressures = data['hourly']['surface_pressure']
        series = list(zip(times, pressures))
        PRESSURE_CACHE[cache_key] = series
        return series
    except Exception:
        PRESSURE_CACHE[cache_key] = None
        return None


def pressure_delta_hpa(series, event_time):
    """
    Calculate pressure change in the 24h window before the event.
    Returns (pressure_at_event, change_24h, flag)
    Negative change = pressure dropped before event.
    """
    if not series:
        return None, None, '—'

    event_str = event_time.strftime('%Y-%m-%dT%H:00')
    minus24_str = (event_time - timedelta(hours=24)).strftime('%Y-%m-%dT%H:00')

    p_event  = None
    p_minus24 = None

    for t, p in series:
        if t == event_str:
            p_event = p
        if t == minus24_str:
            p_minus24 = p

    if p_event is None or p_minus24 is None:
        return p_event, None, '—'

    delta = p_event - p_minus24

    # Flag significant drops — threshold based on published research
    if delta < -8:
        flag = '🌀 PRESSURE DROP'
    elif delta < -4:
        flag = 'pressure falling'
    elif delta > 4:
        flag = 'pressure rising'
    else:
        flag = '—'

    return round(p_event, 1), round(delta, 1), flag


def add_pressure_correlation(df):
    """Add pressure data to cluster events only (to limit API calls)."""
    pressure_hpa   = [None] * len(df)
    pressure_delta = [None] * len(df)
    pressure_flag  = ['—'] * len(df)

    cluster_indices = df.index[df['cluster_flag'] == 'CLUSTER'].tolist()
    if not cluster_indices:
        df['pressure_hpa']   = pressure_hpa
        df['pressure_delta'] = pressure_delta
        df['pressure_flag']  = pressure_flag
        return df

    print(f"  Pressure: fetching for {len(cluster_indices)} cluster events...")
    for i in cluster_indices:
        row = df.loc[i]
        series = fetch_pressure_for_event(row['latitude'], row['longitude'], row['parsed_time'])
        p, delta, flag = pressure_delta_hpa(series, row['parsed_time'])
        pressure_hpa[i]   = p
        pressure_delta[i] = delta
        pressure_flag[i]  = flag

    df['pressure_hpa']   = pressure_hpa
    df['pressure_delta'] = pressure_delta
    df['pressure_flag']  = pressure_flag
    return df
# ─────────────────────────────────────────────────────────────────────────────

# ── HTML Leaflet map ──────────────────────────────────────────────────────────

def write_html(df):
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    total     = len(df)
    clusters  = int((df['cluster_flag'] == 'CLUSTER').sum())
    multi     = int(((df['cluster_flag'] == 'CLUSTER') & (df['network'] == 'MULTI')).sum())
    on_fault  = int((df['fault_proximity'] == 'ON FAULT').sum()) if 'fault_proximity' in df.columns else 0

    markers = []
    for _, row in df.iterrows():
        is_cluster   = row['cluster_flag'] == 'CLUSTER'
        is_precursor = row.get('depth_precursor') == 'SHALLOWING'
        net          = row.get('network', 'USGS')
        colour       = NETWORK_COLOURS.get(net, '#ff3333') if is_cluster else ISOLATED_COLOUR
        radius       = max(4, float(row['mag']) * 3) if is_cluster else max(3, float(row['mag']) * 2)
        depth        = row.get('depth', None)
        depth_str    = f"{depth:.0f}km ({row.get('depth_class','?')})" if pd.notna(depth) else '?'
        dp           = f"{row['dist_prev_km']:.0f}km" if pd.notna(row.get('dist_prev_km')) else '?'
        dn           = f"{row['dist_next_km']:.0f}km" if pd.notna(row.get('dist_next_km')) else '?'
        place        = str(row.get('place', '')).replace("'", "\\'")
        t            = str(row['parsed_time'])[:19]
        warning      = ' ⚠️ SHALLOWING' if is_precursor else ''
        popup = (f"<b>M{row['mag']} — {place}</b><br>"
                 f"Network: {net}<br>"
                 f"Time: {t} UTC<br>"
                 f"Depth: {depth_str}{warning}<br>"
                 f"Prev: {dp} | Next: {dn}<br>"
                 f"Fault: {fault_str_for(row)}<br>"
                 f"Pressure: {row.get('pressure_hpa','?') or '?'} hPa (24h change: {row.get('pressure_delta','?') or '?'} hPa) {row.get('pressure_flag','') or ''}<br>"
                 f"<b>{'🔴 CLUSTER' if is_cluster else '⚪ Isolated'}</b>")
        markers.append({
            'lat':     float(row['latitude']),
            'lon':     float(row['longitude']),
            'colour':  colour,
            'radius':  radius,
            'popup':   popup,
            'cluster': is_cluster,
        })

    markers_json = json.dumps(markers)

    legend_items = ''.join([
        f'<div><span style="background:{c};display:inline-block;width:12px;height:12px;'
        f'border-radius:50%;margin-right:6px;"></span>{n}</div>'
        for n, c in NETWORK_COLOURS.items()
    ])
    legend_items += (f'<div><span style="background:{ISOLATED_COLOUR};display:inline-block;'
                     f'width:12px;height:12px;border-radius:50%;margin-right:6px;"></span>Isolated</div>')

    on_fault_line = f'<div class="on-fault">📍 {on_fault} ON FAULT</div>' if on_fault > 0 else ''
    multi_line    = f'<div class="multi">⭐ {multi} multi-network</div>' if multi > 0 else ''

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Global Seismic Cluster Monitor</title>
  <meta http-equiv="refresh" content="60">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0a0a0f; color:#eee; font-family: 'Helvetica Neue', sans-serif; }}
    #map {{ width:100vw; height:100vh; }}
    #panel {{
      position:absolute; top:16px; left:16px; z-index:1000;
      background:rgba(10,10,20,0.85); border:1px solid #333;
      border-radius:10px; padding:16px; min-width:220px;
      backdrop-filter: blur(8px);
    }}
    #panel h2       {{ font-size:14px; color:#fff; margin-bottom:10px; letter-spacing:1px; }}
    #panel .stat    {{ font-size:22px; font-weight:bold; color:#ff4444; }}
    #panel .label   {{ font-size:11px; color:#999; margin-bottom:8px; }}
    #panel .multi   {{ color:#ffff00; font-size:13px; font-weight:bold; }}
    #panel .on-fault{{ color:#ff8800; font-size:13px; font-weight:bold; }}
    #legend {{ margin-top:12px; border-top:1px solid #333; padding-top:10px; font-size:12px; line-height:22px; }}
    #timestamp {{ font-size:10px; color:#666; margin-top:10px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h2>⚡ SEISMIC CLUSTER MONITOR</h2>
  <div class="stat">{clusters}</div>
  <div class="label">active clusters / {total} global events (24h)</div>
  {multi_line}
  {on_fault_line}
  <div id="legend">{legend_items}</div>
  <div id="timestamp">Updated: {timestamp}<br>Auto-refreshes every 60s</div>
</div>
<script>
  var map = L.map('map', {{ center: [20, 0], zoom: 2, zoomControl: true }});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
  }}).addTo(map);

  var markers = {markers_json};
  markers.forEach(function(m) {{
    L.circleMarker([m.lat, m.lon], {{
      radius:      m.radius,
      fillColor:   m.colour,
      color:       m.cluster ? '#fff' : 'transparent',
      weight:      m.cluster ? 1 : 0,
      opacity:     0.9,
      fillOpacity: m.cluster ? 0.85 : 0.5
    }}).addTo(map).bindPopup(m.popup);
  }});
</script>
</body>
</html>"""

    with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  → Map saved: {HTML_OUTPUT_FILE}")


# ── KML ───────────────────────────────────────────────────────────────────────

def write_kml(df):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '<Document>', f'  <n>Seismic Clusters — {ts}</n>']

    for net, colour in NETWORK_COLOURS.items():
        h = colour.lstrip('#')
        kml_colour = f'ff{h[4:6]}{h[2:4]}{h[0:2]}'
        lines.append(f'''  <Style id="cluster_{net}">
    <IconStyle><color>{kml_colour}</color><scale>1.2</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
    </IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>''')

    lines.append('''  <Style id="isolated">
    <IconStyle><color>ffddaa88</color><scale>0.7</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href></Icon>
    </IconStyle><LabelStyle><scale>0</scale></LabelStyle></Style>''')

    lines.append('  <Folder><n>🔴 Clusters</n>')
    for _, row in df[df['cluster_flag'] == 'CLUSTER'].iterrows():
        net   = row.get('network', 'USGS')
        style = f"cluster_{net}" if net in NETWORK_COLOURS else "cluster_USGS"
        place = str(row.get('place', 'Unknown'))
        dp    = f"{row['dist_prev_km']:.1f}" if pd.notna(row.get('dist_prev_km')) else '?'
        dn    = f"{row['dist_next_km']:.1f}" if pd.notna(row.get('dist_next_km')) else '?'
        pre   = ' ⚠️ SHALLOWING' if row.get('depth_precursor') == 'SHALLOWING' else ''
        lines.append(f'''    <Placemark>
      <n>M{row['mag']} — {place}</n>
      <description>Network: {net}
Time: {str(row['parsed_time'])[:19]} UTC
Magnitude: {row['mag']}  Depth: {row.get('depth','?')} km ({row.get('depth_class','?')}){pre}
Prev: {dp} km | Next: {dn} km
Fault: {fault_str_for(row)}</description>
      <styleUrl>#{style}</styleUrl>
      <Point><coordinates>{row['longitude']},{row['latitude']},0</coordinates></Point>
    </Placemark>''')
    lines.append('  </Folder>')

    lines.append('  <Folder><n>⚪ Isolated</n>')
    for _, row in df[df['cluster_flag'] == '—'].iterrows():
        lines.append(f'''    <Placemark>
      <n>M{row['mag']} — {str(row.get('place',''))}</n>
      <description>Network: {row.get('network','?')}
Time: {str(row['parsed_time'])[:19]} UTC
Magnitude: {row['mag']}  Depth: {row.get('depth','?')} km
Fault: {fault_str_for(row)}</description>
      <styleUrl>#isolated</styleUrl>
      <Point><coordinates>{row['longitude']},{row['latitude']},0</coordinates></Point>
    </Placemark>''')
    lines.append('  </Folder>')
    lines += ['</Document>', '</kml>']

    with open(KML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  → KML saved: {KML_OUTPUT_FILE}")


# ── Report ────────────────────────────────────────────────────────────────────

def report(df):
    total     = len(df)
    clusters  = (df['cluster_flag'] == 'CLUSTER').sum()
    multi     = ((df['cluster_flag'] == 'CLUSTER') & (df['network'] == 'MULTI')).sum()
    precursor = (df['depth_precursor'] == 'SHALLOWING').sum() if 'depth_precursor' in df.columns else 0
    shallow   = df['depth_class'].isin(['very shallow','shallow']).sum() if 'depth_class' in df.columns else 0
    on_fault  = (df['fault_proximity'] == 'ON FAULT').sum() if 'fault_proximity' in df.columns else 0

    print()
    print("═" * 55)
    print(f"  Ground timestamp : {df['parsed_time'].min().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Latest event     : {df['parsed_time'].max().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Total events     : {total}")
    print(f"  Clustered        : {clusters}  ({100*clusters/total:.1f}%)")
    print(f"  Shallow events   : {shallow} (depth < 70km)")
    if on_fault > 0:
        print(f"  📍 On fault      : {on_fault}  ← sitting directly on a known fault")
    if multi > 0:
        print(f"  ⭐ Multi-network : {multi}  ← confirmed by 2+ independent networks")
    pressure_drops = (df['pressure_flag'] == '🌀 PRESSURE DROP').sum() if 'pressure_flag' in df.columns else 0
    if precursor > 0:
        print(f"  ⚠️  Shallowing    : {precursor}  ← depth migration detected")
    if pressure_drops > 0:
        print(f"  🌀 Pressure drop : {pressure_drops}  ← significant pressure fall before event")
    print()

    if clusters > 0:
        print("  Most recent cluster events:")
        for _, row in df[df['cluster_flag'] == 'CLUSTER'].tail(10).iterrows():
            place = str(row.get('place', 'Unknown'))[:35]
            dp    = f"{row['dist_prev_km']:.0f}" if pd.notna(row.get('dist_prev_km')) else '?'
            dn    = f"{row['dist_next_km']:.0f}" if pd.notna(row.get('dist_next_km')) else '?'
            dep   = f"{row['depth']:.0f}km" if pd.notna(row.get('depth')) else '?'
            dc    = row.get('depth_class', '?')
            pre   = ' ⚠️' if row.get('depth_precursor') == 'SHALLOWING' else ''
            fp    = row.get('fault_proximity', '')
            fd    = f" [{fp}]" if fp and fp not in ('unavailable', 'unknown') else ''
            pf = f" {row['pressure_flag']}" if row.get('pressure_flag') and row.get('pressure_flag') not in ('—', None) else ""
            print(f"    [{row.get('network','?'):5}] M{row['mag']:.1f}  {place}  "
                  f"[prev:{dp}km|next:{dn}km] {dep} ({dc}){pre}{fd}{pf}")
    print("═" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Starting live multi-network seismic monitor. Press Ctrl+C to stop.")
    print("Networks: USGS + EMSC + ISC + GFZ + IRIS")
    print("Note: ISC may return 0 events — they publish with 1-2 month delay (gold standard verification)")
    print(f"Outputs : {KML_OUTPUT_FILE}  |  {HTML_OUTPUT_FILE}")
    print(f"Lookback: {LOOKBACK_HOURS}h  |  Refresh: {REFRESH_SECONDS}s\n")

    print("Initialising fault line database...")
    load_fault_lines()
    print()

    while True:
        try:
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] Fetching...")
            raw = fetch_all_networks()
            if len(raw) < 2:
                print("  Not enough data — retrying.")
            else:
                result = run_algorithm(raw)
                report(result)
                write_kml(result)
                write_html(result)
            print(f"  Next refresh in {REFRESH_SECONDS}s...\n")
            time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  Error: {e} — retrying in {REFRESH_SECONDS}s")
            time.sleep(REFRESH_SECONDS)
