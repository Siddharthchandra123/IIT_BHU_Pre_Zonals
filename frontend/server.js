const express = require('express');
const axios = require('axios');
const path = require('path');

const app = express();
const API = "https://ai-chikitsalya.onrender.com"
const PORT = process.env.PORT || 10000;
const PYTHON_API_URL = API;

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render('intro');
});

app.get('/home', (req, res) => {
    res.render('doctor');
});

app.post('/ask', async (req, res) => {
    try {
        const response = await axios.post(`${PYTHON_API_URL}/predict`, req.body);
        res.json({ response: response.data.reply });
    } catch (error) {
        console.error(error.message);
        res.status(500).json({ response: "Backend connection failed" });
    }
});

app.listen(PORT, "0.0.0.0", () => {
    console.log(`✅ Server running on port ${PORT}`);
});