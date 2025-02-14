import { createContext, useContext, useState, useEffect } from 'react';
import { auth } from '@/lib/firebase';
import { 
	signInWithEmailAndPassword,
	createUserWithEmailAndPassword,
	signOut,
	onAuthStateChanged
} from 'firebase/auth';

const AuthContext = createContext({});

export function AuthProvider({ children }) {
	const [user, setUser] = useState(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		const unsubscribe = onAuthStateChanged(auth, (user) => {
			setUser(user);
			setLoading(false);
		});

		return unsubscribe;
	}, []);

	const login = async (email, password) => {
		return signInWithEmailAndPassword(auth, email, password);
	};

	const signup = async (email, password) => {
		return createUserWithEmailAndPassword(auth, email, password);
	};

	const logout = async () => {
		setUser(null);
		await signOut(auth);
	};

	return (
		<AuthContext.Provider value={{ user, login, signup, logout }}>
			{!loading && children}
		</AuthContext.Provider>
	);
}

export const useAuth = () => useContext(AuthContext);