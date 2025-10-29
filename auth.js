// auth.js
require('dotenv').config();
const jwt = require('jsonwebtoken');

const SECRET = process.env.JWT_SECRET || "your_secret_key"; // change this in .env

// verify token middleware
function verifyToken(req, res, next) {
  const token = req.headers['authorization'];
  if (!token) return res.status(401).json({ message: "No token provided" });

  jwt.verify(token.split(" ")[1], SECRET, (err, decoded) => {
    if (err) return res.status(401).json({ message: "Invalid token" });
    req.userId = decoded.id;
    req.role = decoded.role;
    next();
  });
}

// role check middleware
function requireRole(role) {
  return (req, res, next) => {
    if (req.role !== role) return res.status(403).json({ message: "Forbidden" });
    next();
  };
}

module.exports = { verifyToken, requireRole, SECRET };
