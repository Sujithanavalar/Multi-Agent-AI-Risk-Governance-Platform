// Supabase Configuration
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://ckjpcknbroahxjjnnyin.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNranBjbmticm9ha3hqam55aW5wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg2NzkzNjgsImV4cCI6MjA5NDI1NTM2OH0._X0nXFQqk6n7jDRM8v3_myx4ao7sqEa_cqFCYvwpVcE';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
