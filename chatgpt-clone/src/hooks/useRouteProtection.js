import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/components/auth/AuthContext';
import { isPublicRoute, getDefaultPrivateRoute, getDefaultPublicRoute } from '@/config/routes';

export function useRouteProtection() {
	const router = useRouter();
	const { user, loading } = useAuth();
	
	useEffect(() => {
		if (loading) return;

		const currentPath = router.pathname;
		const isPublic = isPublicRoute(currentPath);

		if (!user && !isPublic) {
			// Redirect unauthenticated users to login
			router.push(getDefaultPublicRoute());
		} else if (user && isPublic) {
			// Redirect authenticated users to chat
			router.push(getDefaultPrivateRoute());
		}
	}, [user, loading, router]);

	return { loading };
}