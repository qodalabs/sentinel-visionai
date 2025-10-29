// db.js - Supabase client
require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

// Support values with quotes in .env
const rawUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const rawKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
const SUPABASE_URL = rawUrl.replace(/^"|"$/g, '').trim();
const SUPABASE_ANON_KEY = rawKey.replace(/^"|"$/g, '').trim();

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('❌ Supabase env vars missing: NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY');
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  auth: { persistSession: false }
});

module.exports = supabase;
