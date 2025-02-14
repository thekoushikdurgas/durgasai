import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';

export default function Navigation() {
	const router = useRouter();
	const { user, logout } = useAuth();

	const handleLogout = async () => {
		try {
			await logout();
			router.push('/login');
		} catch (error) {
			console.error('Error logging out:', error);
		}
	};

	const navigationItems = [
		{ path: '/chat', label: 'Chat', icon: '💭' },
		{ path: '/profile', label: 'Profile', icon: '👤' },
		{ path: '/settings', label: 'Settings', icon: '⚙️' },
	];

	if (!user) return null;

	return (
		<nav className="bg-purple-800 p-4">
			<div className="max-w-7xl mx-auto flex justify-between items-center">
				<div className="flex items-center space-x-2">
					<img src="/lucy-logo.png" alt="Lucy™" className="h-8 w-8" />
					<span className="text-white font-bold text-xl">Lucy™</span>
				</div>

				<div className="flex items-center space-x-4">
					{navigationItems.map((item) => (
						<button
							key={item.path}
							onClick={() => router.push(item.path)}
							className={`text-white hover:text-purple-200 px-3 py-2 rounded-md ${
								router.pathname === item.path ? 'bg-purple-700' : ''
							}`}
						>
							<span className="mr-2">{item.icon}</span>
							{item.label}
						</button>
					))}
					<button
						onClick={handleLogout}
						className="text-white hover:text-purple-200 px-3 py-2 rounded-md"
					>
						🚪 Logout
					</button>
				</div>
			</div>
		</nav>
	);
}