import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';

export default function Home() {
	const router = useRouter();
	const { user } = useAuth();

	useEffect(() => {
		if (user) {
			router.push('/chat');
		} else {
			router.push('/login');
		}
	}, [user, router]);

	return (
		<div className="min-h-screen bg-purple-900 flex items-center justify-center">
			<div className="text-white text-xl">Loading...</div>
		</div>
	);

}