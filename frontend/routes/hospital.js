const express = require('express');
const axios = require('axios');
const router = express.Router();

router.get('/hospitals', async (req, res) => {
    const { lat, lon } = req.query;

    try {
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=hospital&limit=10&lat=${lat}&lon=${lon}&radius=50000`;

        const response = await axios.get(url, {
            headers: { 'User-Agent': 'telehealth-app' }
        });

        const hospitals = response.data.map(h => ({
            name: h.display_name.split(",")[0],
            address: h.display_name,
            lat: h.lat,
            lon: h.lon,
            phone: ""
        }));

        res.json(hospitals);

    } catch {
        res.json([]);
    }
});

module.exports = router;