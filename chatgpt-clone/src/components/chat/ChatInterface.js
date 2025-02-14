import { useState, useEffect, useRef } from 'react';
import { db, auth } from '@/lib/firebase';
import { useRouter } from 'next/router';
import { collection, addDoc, query, orderBy, onSnapshot } from 'firebase/firestore';
import { generateResponse } from '@/lib/openrouter';

export default function ChatInterface({ user }) {
	const router = useRouter();
	const [messages, setMessages] = useState([]);
	const [input, setInput] = useState('');
	const [isLoading, setIsLoading] = useState(false);
	const messagesEndRef = useRef(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	useEffect(() => {
		if (!user?.uid) return;
		
		const q = query(
			collection(db, `users/${user.uid}/messages`),
			orderBy('timestamp', 'asc')
		);

		const unsubscribe = onSnapshot(q, (snapshot) => {
			const newMessages = snapshot.docs.map(doc => ({
				id: doc.id,
				...doc.data()
			}));
			setMessages(newMessages);
			scrollToBottom();
		});

		return () => unsubscribe();
	}, [user]);

	const handleSubmit = async (e) => {
		e.preventDefault();
		if (!input.trim() || isLoading) return;

		setIsLoading(true);
		const userMessage = {
			content: input,
			role: 'user',
			timestamp: new Date(),
		};

		try {
			await addDoc(collection(db, `users/${user.uid}/messages`), userMessage);
			setInput('');

			const response = await generateResponse([
				...messages.map(m => ({ role: m.role, content: m.content })),
				{ role: 'user', content: input }
			]);

			const aiMessage = {
				content: response,
				role: 'assistant',
				timestamp: new Date(),
			};

			await addDoc(collection(db, `users/${user.uid}/messages`), aiMessage);
		} catch (error) {
			console.error('Error:', error);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="flex flex-col h-screen bg-purple-900">
			<div className="bg-purple-800 p-4 flex justify-end">
				<div className="flex items-center space-x-4">
					<button
						onClick={() => router.push('/profile')}
						className="text-white hover:text-purple-200 flex items-center space-x-2"
					>
						<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
						</svg>
						<span>Profile</span>
					</button>
					<button
						onClick={() => router.push('/settings')}
						className="text-white hover:text-purple-200 flex items-center space-x-2"
					>
						<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
							<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
						</svg>
						<span>Settings</span>
					</button>
					<button
						onClick={() => auth.signOut()}
						className="text-white hover:text-purple-200"
					>
						Sign Out
					</button>
				</div>
			</div>
			<div className="flex-1 overflow-y-auto p-4 space-y-4">
				{messages.map((message) => (
					<div
						key={message.id}
						className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
					>
						<div
							className={`max-w-[80%] rounded-lg p-3 ${
								message.role === 'user'
									? 'bg-blue-500 text-white'
									: 'bg-purple-800 text-gray-100'
							}`}
						>
							{message.content}
						</div>
					</div>
				))}
				<div ref={messagesEndRef} />
			</div>
			
			<form onSubmit={handleSubmit} className="p-4 border-t border-purple-800">
				<div className="flex space-x-4">
					<input
						type="text"
						value={input}
						onChange={(e) => setInput(e.target.value)}
						placeholder="Type your message..."
						className="flex-1 p-2 rounded-lg bg-purple-800 text-white placeholder-purple-300"
						disabled={isLoading}
					/>
					<button
						type="submit"
						disabled={isLoading}
						className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
					>
						{isLoading ? 'Sending...' : 'Send'}
					</button>
				</div>
			</form>
		</div>
	);
}