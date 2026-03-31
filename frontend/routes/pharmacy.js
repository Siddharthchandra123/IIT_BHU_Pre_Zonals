const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const router = express.Router();

router.get('/pharmacies', async (req, res) => {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
        return res.status(400).json({ error: "Latitude and longitude required." });
    }

    try {
        // ULTRA-SCAN RURAL RADAR (50km Radius + Rural Indian Tags)
        const overpassQuery = `
            [out:json][timeout:25];
            (
              node(around:50000,${lat},${lon})["amenity"~"pharmacy|chemist|doctors|clinic|hospital"];
              node(around:50000,${lat},${lon})["healthcare"~"pharmacy|chemist|centre|dispensary|clinic"];
              node(around:50000,${lat},${lon})["name"~"Medical|Store|Medicos|Dawa|Chemist|Pharmacy|PHC|CHC|Health",i];
              way(around:50000,${lat},${lon})["amenity"~"pharmacy|chemist|doctors|clinic|hospital"];
              way(around:50000,${lat},${lon})["healthcare"~"pharmacy|chemist|centre|dispensary|clinic"];
              way(around:50000,${lat},${lon})["name"~"Medical|Store|Medicos|Dawa|Chemist|Pharmacy|PHC|CHC|Health",i];
            );
            out center;
        `;

        const response = await axios.post('https://overpass-api.de/api/interpreter', 
            `data=${encodeURIComponent(overpassQuery)}`,
            { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
        );
        
        let allResults = [];

        // 1. Map Live OSM Results
        if (response.data && response.data.elements) {
            allResults = response.data.elements.map(p => {
                 const name = p.tags.name || p.tags['name:en'] || "Indian Medical Facility";
                 let addr = p.tags['addr:street'] || p.tags['addr:city'] || "Local Healthcare Point";

                 return {
                     name: name,
                     address: addr,
                     lat: p.lat || (p.center ? p.center.lat : null),
                     lon: p.lon || (p.center ? p.center.lon : null),
                     source: 'satellite'
                 };
            }).filter(p => p.lat && p.lon);
        }

        // 2. Hybrid Merge: Load Local "Data Feed"
        const feedPath = path.join(__dirname, '../pharmacy_data.json');
        if (fs.existsSync(feedPath)) {
            const feedData = JSON.parse(fs.readFileSync(feedPath, 'utf8'));
            // Expanded preview radius (500km) to ensure samples show up for anyone in the same state
            const nearbyFeed = feedData.filter(item => {
                const dist = Math.sqrt(Math.pow(item.lat - parseFloat(lat), 2) + Math.pow(item.lon - parseFloat(lon), 2));
                return dist < 5.0; 
            });
            allResults = [...allResults, ...nearbyFeed];
        }

        res.json(allResults);

    } catch (err) {
        console.error("Overpass API Pharmacy Error:", err.message);
        res.status(500).json({ error: "Failed to connect to satellite radar. Please try again." });
    }
});

module.exports = router;
