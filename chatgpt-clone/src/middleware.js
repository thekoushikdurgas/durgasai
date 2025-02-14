import { NextResponse } from 'next/server';

const publicPaths = ['/login', '/signup', '/forgot-password'];

export function middleware(request) {
	const { pathname } = request.nextUrl;
	const isAuth = request.cookies.get('authenticated');
	const isPublicPath = publicPaths.includes(pathname);

	// Redirect authenticated users away from public paths
	if (isAuth && isPublicPath) {
		return NextResponse.redirect(new URL('/chat', request.url));
	}

	// Redirect unauthenticated users to login
	if (!isAuth && !isPublicPath) {
		return NextResponse.redirect(new URL('/login', request.url));
	}

	return NextResponse.next();
}

export const config = {
	matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};