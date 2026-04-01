const express = require('express');
const fs = require('fs');
const path = require('path');
const router = express.Router();

const bookingsPath = path.join(__dirname, '../opd_bookings.json');
const statusPath = path.join(__dirname, '../serving_status.json');

// Mock Data for selections
const hospitals = [
    { id: "h1", name: "AIIMS - New Delhi", location: "Ansari Nagar" },
    { id: "h2", name: "Safdarjung Hospital", location: "New Delhi" },
    { id: "h3", name: "Max Super Speciality", location: "Saket" }
];

const departments = ["General Medicine", "Cardiology", "Neurology", "Pediatrics"];

const doctors = {
    "General Medicine": [{ id: "d1", name: "Dr. Sharma" }, { id: "d2", name: "Dr. Singh" }],
    "Cardiology": [{ id: "d3", name: "Dr. Verma" }, { id: "d4", name: "Dr. Patel" }],
    "Neurology": [{ id: "d5", name: "Dr. Reddy" }],
    "Pediatrics": [{ id: "d6", name: "Dr. Gupta" }]
};

// GET: Hospitals
router.get('/hospitals', (req, res) => res.json(hospitals));

// GET: Departments
router.get('/departments', (req, res) => res.json(departments));

// POST: Doctors by department
router.post('/doctors', (req, res) => {
    const { dept } = req.body;
    res.json(doctors[dept] || []);
});

// GET: Live Status for a Token
router.get('/status/:tokenId', (req, res) => {
    const { tokenId } = req.params;
    
    if (!fs.existsSync(bookingsPath)) return res.status(404).json({ error: "No bookings found" });
    const bookings = JSON.parse(fs.readFileSync(bookingsPath, 'utf8'));
    const statusData = JSON.parse(fs.readFileSync(statusPath, 'utf8'));

    const myBooking = bookings.find(b => b.id === tokenId);
    if (!myBooking) return res.status(404).json({ error: "Booking not found" });

    const deptStatus = statusData[myBooking.specialty] || { current: "N/A" };
    
    // Calculate position: How many people are before me for this specialty?
    const preceding = bookings.filter(b => 
        b.specialty === myBooking.specialty && 
        b.status !== "Completed" && 
        new Date(b.timestamp) < new Date(myBooking.timestamp)
    );

    // Simplified position for demo: Just count how many UNCOMPLETED bookings match my ID
    // or use a more robust logic
    let position = 0;
    const currentNum = parseInt(deptStatus.current.split('-')[1]) || 0;
    const myNum = parseInt(tokenId.split('-')[1]);
    
    if (myNum > currentNum) {
        position = myNum - currentNum;
    } else {
        position = 0; // Already called or passed
    }

    res.json({
        nowServing: deptStatus.current,
        yourToken: tokenId,
        position: position,
        waitTime: position * 15, // 15 mins per patient
        doctor: deptStatus.doctor
    });
});

module.exports = router;
