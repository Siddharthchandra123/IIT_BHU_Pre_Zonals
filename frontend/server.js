const express = require('express');
const app = express();

console.log("🚀 Server starting...");

app.get('/', (req, res) => {
    res.send("Server is running ✅");
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
    console.log(`✅ Listening on ${PORT}`);
});