const express = require('express');
const axios = require('axios');
const router = express.Router();

router.get('/hospitals', async (req, res) => {
    const { lat, lon } = req.query;

    if (!lat || !lon) {
        return res.status(400).json({ error: "Latitude and longitude required." });
    }

    try {
        // Indian Local Facility Expansion: 
        // 1. Increases radius to 15000m (15km) to cover rural healthcare centers (PHC/CHC)
        // 2. Includes amenity=clinic, doctors, and generic healthcare centers
        // 3. Searches for both point nodes and campus buildings/ways
        const overpassUrl = `https://overpass-api.de/api/interpreter?data=[out:json];(node(around:15000,${lat},${lon})["amenity"~"hospital|clinic|doctors"];node(around:15000,${lat},${lon})["healthcare"~"hospital|clinic|centre|clinic"];way(around:15000,${lat},${lon})["amenity"~"hospital|clinic|doctors"];way(around:15000,${lat},${lon})["healthcare"~"hospital|clinic|centre|clinic"];);out center;`;

        const response = await axios.get(overpassUrl);
        
        if (!response.data || !response.data.elements) {
            return res.json([]);
        }

        const hospitals = response.data.elements.map(h => {
             // Handle Indian naming conventions (PHC, CHC, etc.) and multilingual tags common in India
             const name = h.tags.name || h.tags['name:en'] || h.tags['name:hi'] || h.tags['name:mr'] || "Local Health Centre";
             
             // Extract detailed address components common in Indian datasets
             let addrArray = [];
             if (h.tags['addr:street']) addrArray.push(h.tags['addr:street']);
             if (h.tags['addr:subdistrict']) addrArray.push(h.tags['addr:subdistrict']);
             if (h.tags['addr:district']) addrArray.push(h.tags['addr:district']);
             if (h.tags['addr:city']) addrArray.push(h.tags['addr:city']);
             if (h.tags['addr:state']) addrArray.push(h.tags['addr:state']);

             const addr = addrArray.length > 0 ? addrArray.join(", ") : "Local Medical Facility (Navigate for precise location)";
             
             // Capture any contact numbers stored in various OSM formats
             const phone = h.tags.phone || h.tags['contact:phone'] || h.tags['emergency:phone'] || "";

             return {
                 name: name,
                 address: addr,
                 lat: h.lat || (h.center ? h.center.lat : null),
                 lon: h.lon || (h.center ? h.center.lon : null),
                 phone: phone
             };
        }).filter(h => h.lat && h.lon); // Only return valid GPS points

        res.json(hospitals);

    } catch (err) {
        console.error("Overpass API Indian Extension Error:", err.message);
        // Fallback to empty array instead of crashing on network timeout
        res.json([]);
    }
});

module.exports = router;