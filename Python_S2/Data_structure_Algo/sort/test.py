import folium
from folium.plugins import MarkerCluster

# Verified lab locations with precise coordinates, fun icons, and links
labs = [
    {"name": "NeuroPSI (Saclay)", "lat": 48.7094, "lon": 2.1704, "icon": "flask", "url": "https://neuropsi.cnrs.fr/"},
    {"name": "LNC2 (ENS Paris)", "lat": 48.8431, "lon": 2.3446, "icon": "university", "url": "https://cognition.ens.fr/en"},
    {"name": "UNICOG (NeuroSpin)", "lat": 48.7113, "lon": 2.1474, "icon": "brain", "url": "https://www.unicog.org/"},
    {"name": "LSCP (ENS Paris)", "lat": 48.8431, "lon": 2.3446, "icon": "microscope", "url": "https://lscp.dec.ens.fr/en"},
    {"name": "INCC (Paris Cité)", "lat": 48.8547, "lon": 2.3261, "icon": "book", "url": "https://incc.parisdescartes.fr/"},
    {"name": "VAC (Boulogne-Billancourt)", "lat": 48.8395, "lon": 2.2394, "icon": "puzzle-piece", "url": "https://www.u-paris.fr/laboratoire-voix-audition-cognition-vac/"},
    {"name": "LMC² (Paris)", "lat": 48.8547, "lon": 2.3261, "icon": "cogs", "url": "https://lmc2.parisdescartes.fr/"},
    {"name": "LENA (Pitié-Salpêtrière)", "lat": 48.8373, "lon": 2.3626, "icon": "heartbeat", "url": "https://www.lena.upmc.fr/"}
]

# Create a base map with a dark tile for game style
m = folium.Map(location=[48.8566, 2.3522], zoom_start=11, tiles='CartoDB dark_matter')
marker_cluster = MarkerCluster().add_to(m)

# Add fun ghost-style markers with clickable links
for lab in labs:
    popup_html = f'<a href="{lab["url"]}" target="_blank">{lab["name"]}</a>'
    folium.Marker(
        location=[lab["lat"], lab["lon"]],
        popup=popup_html,
        icon=folium.CustomIcon(
            icon_image='https://cdn-icons-png.flaticon.com/512/3523/3523049.png',
            icon_size=(32, 32)
        )
    ).add_to(marker_cluster)

# Add a custom player icon at center like Pac-Man
folium.Marker(
    location=[48.8566, 2.3522],
    popup='You are here! 🟡',
    icon=folium.CustomIcon(
        icon_image='https://cdn-icons-png.flaticon.com/512/2170/2170773.png',
        icon_size=(36, 36)
    )
).add_to(m)

# Save the map to an HTML file
m.save("game_style_neuro_map.html")
