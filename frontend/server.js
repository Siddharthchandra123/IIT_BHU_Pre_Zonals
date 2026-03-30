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
    res.send("Server is live 🚀");
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
            throw new Error("Missing PYTHON_API_URL");
        }

        const userQuestion = req.body.question;
        const userLang = req.body.lang || "en";

        console.log("➡️ Sending request to backend:", PYTHON_API_URL);

        const response = await axios.post(`${PYTHON_API_URL}/predict`, {
            query: userQuestion,
            lang: userLang
        });

        res.json({ response: response.data.reply });

    } catch (error) {
        console.error("❌ Error:", error.message);
        res.status(500).json({
            response: "Error: Could not connect to AI backend."
        });
    }
});

// ----------- OPTIONAL ROUTES (ADD LATER SAFELY) ----------- //
// Uncomment only after everything works

/*
const hospitalRoutes = require('./routes/hospital');
app.use('/api', hospitalRoutes);
*/

// ----------- START SERVER ----------- //
app.listen(PORT, "0.0.0.0", () => {
    console.log(`✅ Server running on port ${PORT}`);
});