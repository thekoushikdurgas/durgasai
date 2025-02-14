export const routes = {
	public: {
		login: '/login',
		signup: '/signup',
		forgotPassword: '/forgot-password',
		resetPassword: '/reset-password',
	},
	private: {
		chat: '/chat',
		profile: '/profile',
		settings: '/settings',
	}
};

export const navigationLinks = [
	{
		path: routes.private.chat,
		label: 'Chat',
		icon: '💭'
	},
	{
		path: routes.private.profile,
		label: 'Profile',
		icon: '👤'
	},
	{
		path: routes.private.settings,
		label: 'Settings',
		icon: '⚙️'
	}
];

export const isPublicRoute = (path) => {
	return Object.values(routes.public).includes(path);
};

export const getDefaultPrivateRoute = () => routes.private.chat;
export const getDefaultPublicRoute = () => routes.public.login;