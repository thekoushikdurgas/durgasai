import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';
import ChatInterface from '@/components/chat/ChatInterface';
import { auth } from '@/lib/firebase';

export default function Chat() {
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
		<div className="min-h-screen bg-purple-900">
			<nav className="bg-purple-800 p-4 flex justify-between items-center">
				<div className="text-white font-bold text-xl">Lucy™</div>
				<button
					onClick={() => auth.signOut()}
					className="text-white hover:text-purple-200"
				>
					Sign Out
				</button>
			</nav>
			<ChatInterface user={user} />
		</div>
	);
}