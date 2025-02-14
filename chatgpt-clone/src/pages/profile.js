import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';
import ProfileManager from '@/components/profile/ProfileManager';
import Head from 'next/head';

export default function Profile() {
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
				<title>Profile - Lucy™</title>
				<meta name="description" content="Manage your Lucy™ profile" />
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
							onClick={() => router.push('/settings')}
							className="text-white hover:text-purple-200"
						>
							Settings
						</button>
					</div>
				</nav>
				
				<div className="container mx-auto py-8">
					<ProfileManager />
				</div>
			</div>
		</>
	);
}