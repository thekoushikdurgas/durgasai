import { useState } from 'react';
import { useAuth } from '@/components/auth/AuthContext';
import { updateProfile } from 'firebase/auth';
import { db } from '@/lib/firebase';
import { doc, updateDoc } from 'firebase/firestore';

export default function SettingsPanel() {
	const { user } = useAuth();
	const [displayName, setDisplayName] = useState(user?.displayName || '');
	const [theme, setTheme] = useState('dark');
	const [notification, setNotification] = useState(true);
	const [status, setStatus] = useState('');

	const handleSave = async () => {
		try {
			if (user) {
				// Update Firebase Auth profile
				await updateProfile(user, {
					displayName: displayName
				});

				// Update user preferences in Firestore
				const userRef = doc(db, 'users', user.uid);
				await updateDoc(userRef, {
					theme,
					notification,
					lastUpdated: new Date()
				});

				setStatus('Settings saved successfully!');
			}
		} catch (error) {
			setStatus('Error saving settings: ' + error.message);
		}
	};

	return (
		<div className="bg-purple-800 rounded-lg p-6 max-w-2xl mx-auto my-8">
			<h2 className="text-2xl font-bold text-white mb-6">Settings</h2>
			
			<div className="space-y-6">
				<div>
					<label className="block text-white mb-2">Display Name</label>
					<input
						type="text"
						value={displayName}
						onChange={(e) => setDisplayName(e.target.value)}
						className="w-full p-2 rounded bg-purple-900 text-white"
					/>
				</div>

				<div>
					<label className="block text-white mb-2">Theme</label>
					<select
						value={theme}
						onChange={(e) => setTheme(e.target.value)}
						className="w-full p-2 rounded bg-purple-900 text-white"
					>
						<option value="dark">Dark</option>
						<option value="light">Light</option>
					</select>
				</div>

				<div className="flex items-center">
					<input
						type="checkbox"
						checked={notification}
						onChange={(e) => setNotification(e.target.checked)}
						className="mr-2"
					/>
					<label className="text-white">Enable Notifications</label>
				</div>

				{status && (
					<div className={`p-3 rounded ${
						status.includes('Error') ? 'bg-red-500' : 'bg-green-500'
					}`}>
						{status}
					</div>
				)}

				<button
					onClick={handleSave}
					className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
				>
					Save Settings
				</button>
			</div>
		</div>
	);
}