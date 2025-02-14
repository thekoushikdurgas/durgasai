import { useState } from 'react';
import { useAuth } from './AuthContext';
import { useRouter } from 'next/router';
import Image from 'next/image';

export default function SignupForm() {
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState('');
	const { signup } = useAuth();
	const router = useRouter();

	const handleSubmit = async (e) => {
		e.preventDefault();
		try {
			await signup(email, password);
			router.push('/chat');
		} catch (error) {
			setError('Failed to create an account. ' + error.message);
		}
	};

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
					You will be testing one of Lucy™'s core applications:
				</h1>
				<h2 className="text-2xl text-blue-300 mb-8">LUCYideas™</h2>
				<div className="space-x-4">
					<button className="bg-purple-800 text-white px-6 py-2 rounded-lg hover:bg-purple-700">
						What to Expect?
					</button>
					<button className="bg-purple-800 text-white px-6 py-2 rounded-lg hover:bg-purple-700">
						Other Future Applications
					</button>
				</div>
			</div>

			<div className="w-1/2 max-w-md">
				<div className="bg-purple-800 p-8 rounded-lg shadow-xl">
					<h2 className="text-2xl text-white font-bold mb-6">Sign Up to Lucy™</h2>
					{error && <p className="text-red-400 mb-4">{error}</p>}
					<form onSubmit={handleSubmit} className="space-y-6">
						<div>
							<label className="block text-white mb-2">Your Email</label>
							<input
								type="email"
								value={email}
								onChange={(e) => setEmail(e.target.value)}
								className="w-full p-3 rounded-lg bg-purple-900 text-white placeholder-purple-300"
								placeholder="email@example.com"
								required
							/>
						</div>
						<div>
							<label className="block text-white mb-2">Your Password</label>
							<input
								type="password"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								className="w-full p-3 rounded-lg bg-purple-900 text-white placeholder-purple-300"
								placeholder="••••••••••••"
								required
							/>
						</div>
						<button
							type="submit"
							className="w-full bg-blue-400 text-white p-3 rounded-lg hover:bg-blue-500 transition-colors"
						>
							Sign Up
						</button>
					</form>
					<p className="text-white mt-6 text-center">
						Already have an account?{' '}
						<a href="/login" className="text-blue-300 hover:text-blue-200">
							Log In
						</a>
					</p>
				</div>
			</div>
		</div>
	);
}