import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
	apiKey: "AIzaSyA5CfN-iq2myNo9BXh4j0g7GHH39LV1cmY",
	authDomain: "durgas-def2e.firebaseapp.com",
	projectId: "durgas-def2e",
	storageBucket: "durgas-def2e.firebasestorage.app",
	messagingSenderId: "766030362694",
	appId: "1:766030362694:web:725b716d06143cb3a17247"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);