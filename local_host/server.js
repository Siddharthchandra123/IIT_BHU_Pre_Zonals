const express = require('express');
const axios = require('axios');
const app = express();
app.use(express.json());
app.use(express.static('public'));
// Ensure your styles.css is in a 'public' folder
app.set('view engine', 'ejs');

app.get('/', (req, res) => {
    res.render('intro');
    // intro page first
});

app.get('/home', (req, res) => {
    res.render('doctor');  // main app
});

app.get('/ambulance', (req, res) => { res.render('ambulance'); });

app.post('/ask', async (req, res) => {
    try {
        const userQuestion = req.body.question;
        const userLang = req.body.lang || "en";
        // Native Windows to Windows communication        
        const response = await axios.post('http://127.0.0.1:5000/predict', {
            query: userQuestion,
            lang: userLang
        });
        res.json({ response: response.data.reply });
    }
    catch (error) {
        console.error("Connection Error:", error.message);
        res.status(500).json({ response: "Error: Could not connect to the Python AI service." });
    }
});

const hospitalRoutes = require('./routes/hospital');
const pharmacyRoutes = require('./routes/pharmacy');

app.use('/api', hospitalRoutes);
app.use('/api', pharmacyRoutes);

app.get('/hospitals', (req, res) => { res.render('hospital'); });
const fs = require('fs');
const path = require('path');

app.get('/pharmacies', (req, res) => { res.render('pharmacy'); });

// NEW: Data Feed Integration - allows "feeding" local shops
app.post('/api/add-pharmacy', (req, res) => {
    const newPharmacy = req.body;
    const feedPath = path.join(__dirname, 'pharmacy_data.json');
    
    let feedData = [];
    if (fs.existsSync(feedPath)) {
        feedData = JSON.parse(fs.readFileSync(feedPath, 'utf8'));
    }
    
    feedData.push(newPharmacy);
    fs.writeFileSync(feedPath, JSON.stringify(feedData, null, 2));
    
    res.json({ success: true, message: "Pharmacy added to local feed!" });
});

app.listen(3000, () => { console.log('Frontend server live at http://localhost:3000'); });