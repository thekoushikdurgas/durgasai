import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';
import SettingsPanel from '@/components/settings/SettingsPanel';
import Head from 'next/head';

export default function Settings() {
	const { user } = useAuth();
	const router = useRouter();

	useEffect(() => {
		if (!user) {
			router.push('/login');
		}
	}, [user, router]);

	if (!user) {
		return <div className="min-h-screen bg-purple-900 flex items-center justify-center">
			<div className="text-white text-xl">Loading...</div>
		</div>;
	}

	return (
		<>
			<Head>
				<title>Settings - Lucy™</title>
				<meta name="description" content="Manage your Lucy™ account settings" />
			</Head>
			
			<div className="min-h-screen bg-purple-900">
				<nav className="bg-purple-800 p-4 flex justify-between items-center">
					<div className="text-white font-bold text-xl">Lucy™</div>
					<div className="flex space-x-4">
						<button
							onClick={() => router.push('/chat')}
							className="text-white hover:text-purple-200"
						>
							Back to Chat
						</button>
						<button
							onClick={() => router.push('/login')}
							className="text-white hover:text-purple-200"
						>
							Sign Out
						</button>
					</div>
				</nav>
				
				<SettingsPanel />
			</div>
		</>
	);
}