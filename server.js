// server.js
const express = require("express");
const http = require("http");
const cors = require("cors");
const bodyParser = require("body-parser");
const { Server } = require("socket.io");
const path = require('path');
require('dotenv').config();
const supabase = require('./db'); // Supabase client
const jwt = require('jsonwebtoken');
const { verifyToken, requireRole, SECRET } = require('./auth');
const { spawn } = require('child_process');

// Alerts field mapping helpers
const ALERT_SELECT = 'id, weaponType:weapontype, confidence, cameraId:cameraid, imageUrl:imageurl, timestamp';
const toAlertRow = (p) => ({
  weapontype: p.weaponType,
  confidence: p.confidence,
  cameraid: p.cameraId,
  imageurl: p.imageUrl,
  timestamp: p.timestamp,
});
const listAlerts = (search) => {
  let q = supabase.from('alerts').select(ALERT_SELECT).order('timestamp', { ascending: false });
  if (search) q = q.or(`weapontype.ilike.%${search}%,cameraid.ilike.%${search}%`);
  return q;
};

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET","POST","DELETE","PATCH"] }
});

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// =================== ADMIN SECRET ===================
const ADMIN_SECRET = process.env.ADMIN_SECRET || 'be11007';

// =================== SOCKET.IO TOKEN AUTH ===================
io.use((socket, next) => {
  const token = socket.handshake.auth?.token;
  if (!token) return next(new Error("No token provided"));
  jwt.verify(token, SECRET, (err, decoded) => {
    if (err) return next(new Error("Invalid token"));
    socket.user = decoded;
    next();
  });
});

io.on("connection", async (socket) => {
  console.log(`ðŸ”Œ New client connected: ${socket.user?.id || socket.id}`);
  const { data, error } = await listAlerts('');
  if (error) console.error('Supabase fetch alerts error:', error);
  socket.emit("init", data || []);
});

// =================== VIDEO PROCESSING TRACKER ===================
let detectionProcesses = {}; // Track running detection processes by JWT token

// =================== VIDEO PROCESSING PAGE ===================
app.get('/process-video', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'process-video.html'));
});

// =================== VIDEO PROCESSING API ===================
app.post('/api/process-video', verifyToken, async (req, res) => {
  try {
    const mode = req.body.mode;
    if (!mode) return res.status(400).json({ success: false, message: "Specify mode: 'webcam' or 'live'" });

    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ success: false, message: "JWT token missing" });

    if (detectionProcesses[token]) {
      return res.status(400).json({ success: false, message: "Detection already running" });
    }

    const args = ['weapon_detect_5.py'];

    if (mode === 'webcam') {
      args.push('--webcam');
    }

    if (mode === 'live') {
      const ip = req.body.ip || req.body.cctvIp;
      const username = req.body.username || req.body.cctvUsername;
      const password = req.body.password || req.body.cctvPassword;
      const port = req.body.port || req.body.cctvPort;
      const id = req.body.id || req.body.channelId;

      console.log("ðŸ“¡ Live request received:", { ip, username, port, id });

      if (!ip || !username || !password || !port || !id) {
        return res.status(400).json({
          success: false,
          message: "CCTV IP, username, password, port, and channel ID are required"
        });
      }

      const rtspURL = `rtsp://${username}:${password}@${ip}:${port}/Streaming/Channels/${id}`;
      console.log("âœ… RTSP URL built:", rtspURL);

      args.push('--live', rtspURL);
    }

    args.push('--token', token);

    const pyProcess = spawn('python', args, {
      cwd: path.join(__dirname, 'python'),
      // Ensure Python sees both wd-dashboard and yolov5 directories on sys.path
      env: {
        ...process.env,
        PYTHONPATH: [__dirname, path.join(__dirname, 'yolov5')].join(path.delimiter)
      }
    });

    detectionProcesses[token] = pyProcess;

    // Buffer stdout to support line-delimited JSON alerts
    pyProcess._buffer = '';
    pyProcess.stdout.on('data', async (chunk) => {
      pyProcess._buffer += chunk.toString();
      const parts = pyProcess._buffer.split(/\r?\n/);
      pyProcess._buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line) continue;
        try {
          const parsed = JSON.parse(line);
          const row = toAlertRow(parsed);
          const { data: inserted, error } = await supabase
            .from('alerts')
            .insert([row])
            .select(ALERT_SELECT)
            .single();
          if (error) console.error('Supabase insert alert error:', error);
          else io.emit("newAlert", inserted);
        } catch (e) {
          console.log("PYTHON:", line);
        }
      }
    });

    pyProcess.stderr.on('data', (data) => {
      console.error(`PYTHON ERROR: ${data.toString()}`);
    });

    pyProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      if (detectionProcesses[token] === pyProcess) delete detectionProcesses[token];
    });

    res.json({ success: true, message: `Weapon detection started in ${mode} mode.` });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// =================== STOP DETECTION API (Webcam + Live) ===================
app.post('/api/stop-detection', verifyToken, async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    const proc = detectionProcesses[token];
    if (proc) {
      proc.kill('SIGINT');
      delete detectionProcesses[token];
      return res.json({ success: true, message: "Detection stopped successfully" });
    }
    res.status(400).json({ success: false, message: "No running detection process" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// =================== ALERT ROUTES ===================
app.get("/api/alerts", verifyToken, async (req, res) => {
  try {
    const search = (req.query.search || '').trim();
    const { data, error } = await listAlerts(search);
    if (error) throw error;
    res.json(data);
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.get("/api/admin-alerts", verifyToken, requireRole('admin'), async (req, res) => {
  try {
    const search = (req.query.search || '').trim();
    const { data, error } = await listAlerts(search);
    if (error) throw error;
    res.json(data);
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.post("/api/alerts", verifyToken, async (req, res) => {
  try {
    const payload = { ...req.body };
    const row = toAlertRow(payload);
    const { data, error } = await supabase
      .from('alerts')
      .insert([row])
      .select(ALERT_SELECT)
      .single();
    if (error) throw error;
    io.emit("newAlert", data);
    res.json({ success: true, alert: data });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.delete("/api/alerts/:id", verifyToken, requireRole('admin'), async (req, res) => {
  try {
    const { data: deleted, error } = await supabase.from('alerts').delete().eq('id', req.params.id).select().maybeSingle();
    if (error) throw error;
    if (!deleted) return res.status(404).json({ success: false, message: "Alert not found" });
    const { data: refreshed } = await supabase.from('alerts').select(ALERT_SELECT).order('timestamp', { ascending: false });
    io.emit("init", refreshed || []);
    res.json({ success: true, message: "Alert deleted" });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// =================== AUTH ROUTES ===================
app.post('/api/register', async (req, res) => {
  try {
    const { username, password, role, adminSecret } = req.body;
    if (!username || !password) return res.status(400).json({ success: false, message: 'Username and password required' });

    if (role === 'admin' && adminSecret !== ADMIN_SECRET) {
      return res.status(403).json({ success: false, message: "Invalid admin secret password" });
    }

    const bcrypt = require('bcrypt');
    const hashed = await bcrypt.hash(password, 10);

    // Enforce unique username
    const { data: existing, error: exErr } = await supabase.from('users').select('id').eq('username', username).maybeSingle();
    if (exErr) throw exErr;
    if (existing) return res.status(409).json({ success: false, message: 'Username already exists' });

    const { error } = await supabase.from('users').insert([{ username, password: hashed, role: role || 'user' }]);
    if (error) throw error;
    res.json({ success: true, message: "User registered successfully" });
  } catch (err) {
    res.status(400).json({ success: false, message: err.message });
  }
});

app.post('/api/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    const { data: user, error } = await supabase.from('users').select('id, username, password, role').eq('username', username).single();
    if (error) return res.status(400).json({ success: false, message: "User not found" });
    const bcrypt = require('bcrypt');
    const valid = await bcrypt.compare(password, user.password);
    if (!valid) return res.status(400).json({ success: false, message: "Wrong password" });
    const token = jwt.sign({ id: user.id, role: user.role, username: user.username }, SECRET, { expiresIn: '1h' });
    res.json({ success: true, token, role: user.role });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.get('/api/admin-data', verifyToken, requireRole('admin'), (req, res) => {
  res.json({ secret: "This is admin-only data" });
});

// =================== USER MANAGEMENT ROUTES ===================
app.get('/api/admin-users', verifyToken, requireRole('admin'), async (req, res) => {
  try {
    const { data, error } = await supabase.from('users').select('id, username, role').order('username');
    if (error) throw error;
    res.json(data);
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

app.delete('/api/users/:id', verifyToken, requireRole('admin'), async (req, res) => {
  try {
    const { data, error } = await supabase.from('users').delete().eq('id', req.params.id).select().maybeSingle();
    if (error) return res.status(404).json({ success: false, message: "User not found" });
    res.json({ success: true, message: "User deleted" });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// =================== SERVE STATIC FILES ===================
app.use(express.static(path.join(__dirname, "public")));
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// =================== HEALTH CHECK ===================
app.get("/health", (req, res) => res.json({ ok: true, message: "backend up" }));

// =================== START SERVER ===================
const PORT = 5000;
server.listen(PORT, () => {
  console.log(`ðŸš€ Backend running on http://localhost:${PORT}`);
});


