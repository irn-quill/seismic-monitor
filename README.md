# Global Seismic Cluster Monitor

**What It Does**
This is a live global seismic cluster detection system. It pulls earthquake data from five independent international networks every five minutes, cross-checks them against each other, and plots the results on an interactive world map. Red dots are clusters — earthquakes occurring close together in space and time. The brighter they are, the more networks agree.

**The Five Networks**
USGS (Americas, real-time) · EMSC (Europe, Mediterranean) · ISC (global gold standard, verified) · GFZ Potsdam (Asia, Central Europe) · IRIS/EarthScope (ocean floor). Each network is independent. When two or more networks flag the same event as part of a cluster, it appears in yellow — a MULTI-network confirmation.

**The Algorithm**
For each earthquake, the system checks both its predecessor and successor in time. If either neighbour is within 100km and 1 hour, the event is flagged as part of a cluster. This bidirectional check catches patterns that backwards-only systems miss — validated against 9,892 USGS events where it found 1,381 clusters the original algorithm couldn't see.
The algorithm also tracks whether clusters are getting shallower over time (a known precursor pattern), calculates proximity to the GEM Global Active Faults Database (161,622 segments), and correlates atmospheric pressure changes in the 24 hours before each event.

**What It Has Found**
On its first full day of operation (27 February 2026) the system independently traced the Ring of Fire from clustering patterns alone, identified Hawaii as an intraplate hotspot by the absence of nearby events, flagged the Bangladesh earthquake during the same refresh cycle as international news reporting, and detected Permian Basin fracking clusters in West Texas without any knowledge of oil extraction activity.

**What It Does Not Claim**
This system does not predict earthquakes. Cluster detection is observational. Depth shallowing and pressure correlation are flagged as signals for human interpretation, not automated forecasts. The algorithm contains no hardcoded geology — everything it finds, it finds from the data.

**Technical**
~420 lines of Python. No external seismology libraries. Runs on GitHub Actions, free tier. Updates every 5 minutes. Source code in this repository.

Live map:
https://irn-quill.github.io/seismic-monitor/seismic_map.html

Live data:
https://irn-quill.github.io/seismic-monitor/seismic_data.html
