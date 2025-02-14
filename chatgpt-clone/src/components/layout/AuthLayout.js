import Image from 'next/image';
import { useRouter } from 'next/router';

export default function AuthLayout({ children }) {
	const router = useRouter();
	const isLoginPage = router.pathname === '/login';

	return (
		<div className="min-h-screen bg-purple-900 flex items-center justify-between p-8">
			<div className="w-1/2 pr-8">
				<div className="mb-8">
					<Image
						src="/lucy-logo.png"
						alt="Lucy™"
						width={150}
						height={150}
						className="mb-4"
					/>
				</div>
				<h1 className="text-4xl font-bold text-white mb-4">
					Welcome to Lucy™
				</h1>
				<h2 className="text-2xl text-blue-300 mb-8">
					Innovation Starts Here
				</h2>
				<div className="space-x-4">
					<button 
						onClick={() => router.push('/about')}
						className="bg-purple-800 text-white px-6 py-2 rounded-lg hover:bg-purple-700"
					>
						Learn More
					</button>
				</div>
			</div>

			<div className="w-1/2 max-w-md">
				<div className="bg-purple-800 p-8 rounded-lg shadow-xl">
					{children}
				</div>
			</div>
		</div>
	);
}