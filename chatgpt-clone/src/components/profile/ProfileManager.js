import { useState, useEffect } from 'react';
import { useAuth } from '@/components/auth/AuthContext';
import { db } from '@/lib/firebase';
import { doc, getDoc, setDoc } from 'firebase/firestore';
import Image from 'next/image';

export default function ProfileManager() {
	const { user } = useAuth();
	const [profile, setProfile] = useState({
		displayName: '',
		bio: '',
		preferences: {
			language: 'en',
			modelPreference: 'gpt-4'
		}
	});
	const [isLoading, setIsLoading] = useState(true);
	const [isSaving, setIsSaving] = useState(false);
	const [message, setMessage] = useState('');

	useEffect(() => {
		if (user) {
			loadProfile();
		}
	}, [user]);

	const loadProfile = async () => {
		try {
			const docRef = doc(db, 'users', user.uid);
			const docSnap = await getDoc(docRef);
			
			if (docSnap.exists()) {
				setProfile(docSnap.data());
			}
		} catch (error) {
			console.error('Error loading profile:', error);
			setMessage('Failed to load profile');
		} finally {
			setIsLoading(false);
		}
	};

	const handleSave = async () => {
		setIsSaving(true);
		try {
			await setDoc(doc(db, 'users', user.uid), profile);
			setMessage('Profile updated successfully!');
		} catch (error) {
			console.error('Error saving profile:', error);
			setMessage('Failed to save profile');
		} finally {
			setIsSaving(false);
		}
	};

	if (isLoading) {
		return <div className="text-white text-center">Loading profile...</div>;
	}

	return (
		<div className="max-w-2xl mx-auto p-6 bg-purple-800 rounded-lg shadow-xl">
			<div className="flex items-center space-x-4 mb-6">
				<div className="relative w-20 h-20">
					<Image
						src={user.photoURL || '/default-avatar.png'}
						alt="Profile"
						layout="fill"
						className="rounded-full"
					/>
				</div>
				<div>
					<h2 className="text-2xl font-bold text-white">{user.email}</h2>
					<p className="text-purple-300">Member since {new Date(user.metadata.creationTime).toLocaleDateString()}</p>
				</div>
			</div>

			<div className="space-y-4">
				<div>
					<label className="block text-white mb-2">Display Name</label>
					<input
						type="text"
						value={profile.displayName}
						onChange={(e) => setProfile({...profile, displayName: e.target.value})}
						className="w-full p-2 rounded bg-purple-900 text-white"
					/>
				</div>

				<div>
					<label className="block text-white mb-2">Bio</label>
					<textarea
						value={profile.bio}
						onChange={(e) => setProfile({...profile, bio: e.target.value})}
						className="w-full p-2 rounded bg-purple-900 text-white h-24"
					/>
				</div>

				<div>
					<label className="block text-white mb-2">Preferred Language</label>
					<select
						value={profile.preferences.language}
						onChange={(e) => setProfile({
							...profile,
							preferences: {...profile.preferences, language: e.target.value}
						})}
						className="w-full p-2 rounded bg-purple-900 text-white"
					>
						<option value="en">English</option>
						<option value="es">Spanish</option>
						<option value="fr">French</option>
					</select>
				</div>

				{message && (
					<div className={`p-3 rounded ${
						message.includes('success') ? 'bg-green-500' : 'bg-red-500'
					} text-white`}>
						{message}
					</div>
				)}

				<button
					onClick={handleSave}
					disabled={isSaving}
					className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:opacity-50"
				>
					{isSaving ? 'Saving...' : 'Save Profile'}
				</button>
			</div>
		</div>
	);
}