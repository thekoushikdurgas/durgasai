import { useState } from 'react';
import { useAuth } from '@/components/auth/AuthContext';
import { useRouter } from 'next/router';

export default function Login() {
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState('');
	const { login } = useAuth();
	const router = useRouter();

	const handleSubmit = async (e) => {
		e.preventDefault();
		try {
			await login(email, password);
			router.push('/');
		} catch (error) {
			setError('Failed to login. Please check your credentials.');
		}
	};

	return (
		<div className="min-h-screen bg-purple-700 flex items-center justify-center">
			<div className="bg-purple-800 p-8 rounded-lg shadow-xl w-96">
				<h2 className="text-2xl text-white font-bold mb-6">Log In to Lucy™</h2>
				{error && <p className="text-red-400 mb-4">{error}</p>}
				<form onSubmit={handleSubmit}>
					<div className="mb-4">
						<label className="block text-white mb-2">Your Email</label>
						<input
							type="email"
							value={email}
							onChange={(e) => setEmail(e.target.value)}
							className="w-full p-2 rounded bg-purple-900 text-white"
							placeholder="email@example.com"
						/>
					</div>
					<div className="mb-6">
						<label className="block text-white mb-2">Your Password</label>
						<input
							type="password"
							value={password}
							onChange={(e) => setPassword(e.target.value)}
							className="w-full p-2 rounded bg-purple-900 text-white"
						/>
					</div>
					<button
						type="submit"
						className="w-full bg-blue-400 text-white p-2 rounded hover:bg-blue-500"
					>
						Log In
					</button>
				</form>
			</div>
		</div>
	);
}