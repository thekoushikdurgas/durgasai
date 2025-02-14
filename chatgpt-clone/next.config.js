/** @type {import('next').NextConfig} */
const nextConfig = {
	images: {
		domains: ['api.openrouter.ai'],
	},
	env: {
		NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
	},
};

module.exports = nextConfig;