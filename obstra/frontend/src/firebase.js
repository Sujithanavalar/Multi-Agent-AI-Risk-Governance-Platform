// Firebase Configuration
// You need to create a Firebase project at https://console.firebase.google.com/
// Then replace these values with your actual Firebase config!
// See https://firebase.google.com/docs/web/setup for more info

import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Replace this config object with your own from Firebase Console
const firebaseConfig = {
  apiKey: "AIzaSyAGBuS7Y7WKYBsNhRnlX0GvuY7KGsW4f2U",
  authDomain: "obstra.firebaseapp.com",
  projectId: "obstra",
  storageBucket: "obstra.firebasestorage.app",
  messagingSenderId: "894785483193",
  appId: "1:894785483193:web:e06cb604bb6530e1d28668",
  measurementId: "G-K903DLXJNJ"
};

// Initialize Firebase only if it's not already initialized
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
export const auth = getAuth(app);
export const db = getFirestore(app);

export default app;
