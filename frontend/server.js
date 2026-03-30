const express = require('express');
const axios = require('axios');
const path = require('path');
const ejs = require('ejs'); // Explicitly link EJS at startup
const app = express();

// Production-ready Cloud Configuration
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:5000';

// Explicitly set paths for views and static files to avoid Linux path errors
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public'))); 

app.get('/', (req, res) => {
    res.render('intro');   // intro page first
});

app.get('/home', (req, res) => {
    res.render('doctor');  // main app
});

app.get('/ambulance', (req, res) => {
    res.render('ambulance');
});

app.post('/ask', async (req, res) => {
    try {
        const userQuestion = req.body.question;
        const userLang = req.body.lang || "en";

        // Dynamically connecting to the AI Backend URL (Local or Cloud)
        const response = await axios.post(`${PYTHON_API_URL}/predict`, {
            query: userQuestion,
            lang: userLang
        });

        res.json({ response: response.data.reply });
    } catch (error) {
        console.error("Connection Error:", error.message);
        res.status(500).json({ response: "Error: Could not connect to the Python AI service." });
    }
});

const hospitalRoutes = require('./routes/hospital');
app.use('/api', hospitalRoutes);

app.get('/hospitals', (req, res) => {
    res.render('hospital');
});

app.listen(PORT, () => {
    console.log(`Frontend server live at port ${PORT}`);
});