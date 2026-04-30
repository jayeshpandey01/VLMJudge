/**
 * Name: Jayesh Pandey
 * Summary: Source file for config.js in the firebase module.
 */

import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCOpxVZ68e8H3_wA5E59mEvizaunIe3-1Q",
  authDomain: "trainlq.firebaseapp.com",
  projectId: "trainlq",
  storageBucket: "trainlq.firebasestorage.app",
  messagingSenderId: "1031179454445",
  appId: "1:1031179454445:web:a5013a5b19e70dea8709e4",
  measurementId: "G-EWFWYW20QL"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export default app;
