const express = require('express');
const axios = require('axios');
const path = require('path');

const app = express();

console.log("🚀 Server starting...");

// ----------- CONFIG ----------- //
const PORT = process.env.PORT || 10000;
const PYTHON_API_URL = process.env.PYTHON_API_URL;

// ----------- MIDDLEWARE ----------- //
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static files
app.use(express.static(path.join(__dirname, 'public')));

// EJS setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

// ----------- ROUTES ----------- //

// Health check (important for Render)
app.get('/', (req, res) => {
    res.render('intro');
});
// Intro page
app.get('/intro', (req, res) => {
    res.render('intro');
});

// Main app page
app.get('/home', (req, res) => {
    res.render('doctor');
});

// Ambulance page
app.get('/ambulance', (req, res) => {
    res.render('ambulance');
});

// Hospital page
app.get('/hospitals', (req, res) => {
    res.render('hospital');
});

// AI request route
app.post('/ask', async (req, res) => {
    try {
        if (!PYTHON_API_URL) {
            throw new Error("Missing PYTHON_API_URL in Environment Variables");
        }

        const userQuestion = req.body.question;
        const userLang = req.body.lang || "en";
        
        // Strip trailing slash if the user accidentally added it in Render config
        const safeApiUrl = PYTHON_API_URL.endsWith('/') ? PYTHON_API_URL.slice(0, -1) : PYTHON_API_URL;

        console.log(`➡️ Sending request to backend: ${safeApiUrl}/predict`);

        const response = await axios.post(`${safeApiUrl}/predict`, {
            query: userQuestion,
            lang: userLang
        }, { timeout: 60000 }); // 60s timeout to allow cold start

        res.json({ response: response.data.reply });

    } catch (error) {
        console.error("❌ Error:", error.message);
        if (error.response) {
            console.error("Backend returned:", error.response.data);
        }
        res.status(500).json({
            response: `Error: Could not connect to AI backend. Reason: ${error.message}`
        });
    }
});

// ----------- API ROUTES ----------- //
const hospitalRoutes = require('./routes/hospital');
const pharmacyRoutes = require('./routes/pharmacy');
const opdRoutes = require('./routes/opd');
const patientRoutes = require('./routes/patient');

app.use('/api', hospitalRoutes);
app.use('/api', pharmacyRoutes);
app.use('/api/opd', opdRoutes);
app.use('/api/patient', patientRoutes);

// View routes
app.get('/hospitals', (req, res) => { res.render('hospital'); });
app.get('/pharmacies', (req, res) => { res.render('pharmacy'); });
app.get('/opd', (req, res) => { res.render('opd'); });

// NEW: Data Feed Integration - allows "feeding" local shops in production
app.post('/api/add-pharmacy', (req, res) => {
    const fs = require('fs');
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

// ----------- START SERVER ----------- //
app.listen(PORT, "0.0.0.0", () => {
    console.log(`✅ Server running on port ${PORT}`);
});