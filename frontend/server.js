const express = require('express');
const axios = require('axios');
const app = express();

// Production-ready Cloud Configuration
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:5000';

app.use(express.json());
app.use(express.static('public')); // Ensure your styles.css is in a 'public' folder
app.set('view engine', 'ejs');

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