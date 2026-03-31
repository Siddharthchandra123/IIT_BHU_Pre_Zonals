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
        // 2. Includes amenity=hospital, clinic, doctors, dentist
        // 3. Searches for both point nodes and campus buildings/ways
        const overpassUrl = `https://overpass-api.de/api/interpreter?data=[out:json];(node(around:15000,${lat},${lon})["amenity"~"hospital|clinic|doctors|dentist"];node(around:15000,${lat},${lon})["healthcare"~"hospital|clinic|centre|dentist|surgeon"];way(around:15000,${lat},${lon})["amenity"~"hospital|clinic|doctors|dentist"];way(around:15000,${lat},${lon})["healthcare"~"hospital|clinic|centre|dentist|surgeon"];);out center;`;

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

             const addr = addrArray.length > 0 ? addrArray.join(", ") : "Local Medical Facility";
             
             // Capture any contact numbers stored in various OSM formats
             const phone = h.tags.phone || h.tags['contact:phone'] || h.tags['emergency:phone'] || "";

             // --- Classification Logic ---
             let category = "General";
             let icon = "🏥";

             const type = (h.tags.amenity || h.tags.healthcare || "").toLowerCase();
             const speciality = (h.tags['healthcare:speciality'] || "").toLowerCase();

             if (type.includes('dentist') || speciality.includes('dentist')) {
                 category = "Dentistry";
                 icon = "🦷";
             } else if (type.includes('surgeon') || speciality.includes('surgery') || speciality.includes('surgeon')) {
                 category = "Surgeon";
                 icon = "🔬";
             } else if (type.includes('clinic')) {
                 category = "Clinic";
                 icon = "🩺";
             } else if (type.includes('hospital')) {
                 category = "Multispeciality";
                 icon = "🏥";
             }

             return {
                 name: name,
                 address: addr,
                 category: category,
                 icon: icon,
                 lat: h.lat || (h.center ? h.center.lat : null),
                 lon: h.lon || (h.center ? h.center.lon : null),
                 phone: phone
             };
        }).filter(h => h.lat && h.lon); // Only return valid GPS points

        res.json(hospitals);

    } catch (err) {
        console.error("Overpass API Indian Extension Error:", err.message);
        res.json([]);
    }
});

module.exports = router;