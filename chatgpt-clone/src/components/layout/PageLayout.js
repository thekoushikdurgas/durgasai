import Head from 'next/head';

export default function PageLayout({ children, title, description }) {
	return (
		<>
			<Head>
				<title>{title ? `${title} - Lucy™` : 'Lucy™'}</title>
				<meta name="description" content={description || 'Lucy™ - Innovation Starts Here'} />
				<meta name="viewport" content="width=device-width, initial-scale=1" />
				<link rel="icon" href="/favicon.ico" />
			</Head>

			<main className="container mx-auto px-4 py-8">
				{children}
			</main>

			<footer className="bg-purple-800 text-white text-center py-4 mt-auto">
				<p>&copy; {new Date().getFullYear()} Lucy™. All rights reserved.</p>
			</footer>
		</>
	);
}