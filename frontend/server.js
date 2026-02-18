const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());
app.use(express.static('public')); // Ensure your styles.css is in a 'public' folder
app.set('view engine', 'ejs');
app.get('/', (req, res) => {
    res.render('intro');   // intro page first
});

app.get('/home', (req, res) => {
    res.render('doctor');  // main app
});

app.post('/ask', async (req, res) => {
    try {
        const userQuestion = req.body.question;
        
        // Native Windows to Windows communication
        const response = await axios.post('http://127.0.0.1:5000/predict', {
            query: userQuestion
        });

        res.json({ response: response.data.reply });
    } catch (error) {
        console.error("Connection Error:", error.message);
        res.status(500).json({ response: "Error: Could not connect to the Python AI service." });
    }
});

app.listen(3000, () => {
    console.log('Frontend server live at http://localhost:3000');
});