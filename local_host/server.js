const express = require('express');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();

console.log("🚀 Local Server starting...");

// ----------- CONFIG ----------- //
const PORT = process.env.PORT || 3000;
// For local_host, the Python API is always running on 5000
const PYTHON_API_URL = "http://127.0.0.1:5000";

// ----------- MIDDLEWARE ----------- //
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static files
app.use(express.static(path.join(__dirname, 'public')));

// EJS setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

// ----------- ROUTES ----------- //

// Main Landing
app.get('/', (req, res) => {
    res.render('intro');
});

// Intro explicitly
app.get('/intro', (req, res) => {
    res.render('intro');
});

// AI Assistant page
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

// Pharmacy page
app.get('/pharmacies', (req, res) => {
    res.render('pharmacy');
});

// OPD Booking page
app.get('/opd', (req, res) => {
    res.render('opd');
});

// AI request route proxy
app.post('/ask', async (req, res) => {
    try {
        const userQuestion = req.body.question;
        const userLang = req.body.lang || "en";
        
        console.log(`➡️ Proxying request to Python API: ${PYTHON_API_URL}/predict`);

        const response = await axios.post(`${PYTHON_API_URL}/predict`, {
            query: userQuestion,
            lang: userLang
        }, { timeout: 30000 });

        res.json({ response: response.data.reply });

    } catch (error) {
        console.error("❌ Proxy Error:", error.message);
        res.status(500).json({
            response: `Error: AI backend unavailable. Ensure Python API.py is running on port 5000.\n(${error.message})`
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

// New: Data Feed Integration for manual data entry
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

// ----------- START SERVER ----------- //
app.listen(PORT, "0.0.0.0", () => {
    console.log(`✅ Local host server running at http://localhost:${PORT}`);
    console.log(`📡 Connected to Python API at ${PYTHON_API_URL}`);
});