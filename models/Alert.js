const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  weaponType: String,
  confidence: Number,
  cameraId: String,
  timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Alert', alertSchema);
