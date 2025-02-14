// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
	apiKey: "AIzaSyA5CfN-iq2myNo9BXh4j0g7GHH39LV1cmY",
	authDomain: "durgas-def2e.firebaseapp.com",
	projectId: "durgas-def2e",
	storageBucket: "durgas-def2e.firebasestorage.app",
	messagingSenderId: "766030362694",
	appId: "1:766030362694:web:725b716d06143cb3a17247"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

export { db, auth };