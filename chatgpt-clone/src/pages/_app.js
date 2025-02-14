import '@/styles/globals.css';
import { AuthProvider } from '@/components/auth/AuthContext';
import Navigation from '@/components/navigation/Navigation';
import { useRouter } from 'next/router';

const publicPaths = ['/login', '/signup', '/forgot-password'];

function MyApp({ Component, pageProps }) {
	const router = useRouter();
	const isPublicPath = publicPaths.includes(router.pathname);

	return (
		<AuthProvider>
			<div className="min-h-screen bg-purple-900">
				{!isPublicPath && <Navigation />}
				<Component {...pageProps} />
			</div>
		</AuthProvider>
	);
}

export default MyApp;
