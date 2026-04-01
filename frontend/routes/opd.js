const express = require('express');
const fs = require('fs');
const path = require('path');
const router = express.Router();

const bookingsPath = path.join(__dirname, '../opd_bookings.json');

// Helper to read bookings
const readBookings = () => {
    if (!fs.existsSync(bookingsPath)) {
        return [];
    }
    const data = fs.readFileSync(bookingsPath, 'utf8');
    return JSON.parse(data);
};

// Helper to write bookings
const writeBookings = (bookings) => {
    fs.writeFileSync(bookingsPath, JSON.stringify(bookings, null, 2));
};

// POST: Book a new OPD token
router.post('/book', (req, res) => {
    const { name, phone, specialty, date, hospital, doctor } = req.body;

    if (!name || !phone || !specialty || !date) {
        return res.status(400).json({ success: false, message: "All fields are required." });
    }

    const bookings = readBookings();
    
    // Generate a new token ID (sequential)
    const lastId = bookings.length > 0 ? parseInt(bookings[bookings.length - 1].id.split('-')[1]) : 1000;
    const newId = `TK-${lastId + 1}`;

    const newBooking = {
        id: newId,
        name,
        phone,
        specialty,
        date,
        hospital: hospital || "Max Super Speciality", // Default for demo
        doctor: doctor || "Dr. Sharma", // Default for demo
        status: "Confirmed",
        timestamp: new Date().toISOString()
    };

    bookings.push(newBooking);
    writeBookings(bookings);

    res.json({ success: true, token: newId, booking: newBooking });
});

// GET: Admin next token simulation
router.post('/admin/next', (req, res) => {
    const { dept } = req.body;
    const statusPath = path.join(__dirname, '../serving_status.json');
    const statusData = JSON.parse(fs.readFileSync(statusPath, 'utf8'));

    if (statusData[dept]) {
        const currentId = statusData[dept].current;
        const currentNum = parseInt(currentId.split('-')[1]);
        statusData[dept].current = `TK-${currentNum + 1}`;
        statusData[dept].lastCalled = new Date().toISOString();
        fs.writeFileSync(statusPath, JSON.stringify(statusData, null, 2));
        res.json({ success: true, next: statusData[dept].current });
    } else {
        res.status(404).json({ error: "Department not found" });
    }
});

// POST: Update booking status to 'Paid'
router.post('/pay', (req, res) => {
    const { tokenId } = req.body;

    if (!tokenId) {
        return res.status(400).json({ success: false, message: "Token ID is required." });
    }

    const bookings = readBookings();
    const index = bookings.findIndex(b => b.id === tokenId);

    if (index === -1) {
        return res.status(404).json({ success: false, message: "Booking not found." });
    }

    bookings[index].status = "Paid";
    writeBookings(bookings);

    res.json({ success: true, message: "Payment successful!" });
});

// GET: List all bookings (for demo view)
router.get('/list', (req, res) => {
    const bookings = readBookings();
    res.json(bookings);
});

module.exports = router;
