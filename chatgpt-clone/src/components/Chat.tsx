'use client';

import { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { auth, db } from '@/lib/firebase';
import { signOut } from 'firebase/auth';
import { collection, addDoc, query, orderBy, onSnapshot } from 'firebase/firestore';
import Image from 'next/image';

interface Message {
	id?: string;
	content: string;
	role: 'user' | 'assistant';
	timestamp: Date;
}

export default function Chat() {
	const { user } = useAuth();
	const [input, setInput] = useState('');
	const [messages, setMessages] = useState<Message[]>([]);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const messagesEndRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (user) {
			const q = query(
				collection(db, `users/${user.uid}/messages`),
				orderBy('timestamp', 'asc')
			);
			
			const unsubscribe = onSnapshot(q, (snapshot) => {
				const msgs = snapshot.docs.map(doc => ({
					id: doc.id,
					...doc.data()
				} as Message));
				setMessages(msgs);
			});

			return () => unsubscribe();
		}
	}, [user]);

	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSendMessage();
		}
	};

	const handleSendMessage = async () => {
		if (!input.trim() || !user || isLoading) return;

		setIsLoading(true);
		setError(null);
		const userMessage: Message = {
			content: input,
			role: 'user',
			timestamp: new Date(),
		};

		try {
			await addDoc(collection(db, `users/${user.uid}/messages`), userMessage);
			setInput('');

			const response = await fetch('/api/chat', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ message: input }),
			});

			const data = await response.json();
			
			if (data.error) {
				throw new Error(data.error);
			}
			
			if (data.message) {
				const assistantMessage: Message = {
					content: data.message,
					role: 'assistant',
					timestamp: new Date(),
				};
				await addDoc(collection(db, `users/${user.uid}/messages`), assistantMessage);
			}
		} catch (error: any) {
			console.error('Error sending message:', error);
			setError(error.message || 'Failed to send message');
		} finally {
			setIsLoading(false);
		}
	};

	const handleLogout = async () => {
		try {
			await signOut(auth);
		} catch (error) {
			console.error('Error signing out:', error);
		}
	};

	return (
		<div className="flex flex-col h-screen bg-gray-50">
			<div className="flex-none bg-white border-b shadow-sm">
				<div className="max-w-6xl mx-auto px-4 py-3">
					<div className="flex justify-between items-center">
						<div className="flex items-center space-x-3">
							<Image
								src="/durgas-logo.svg"
								alt="Durgas AI Logo"
								width={40}
								height={40}
								className="w-10 h-10"
							/>
							<h1 className="text-xl font-semibold text-gray-800">Durgas AI</h1>
						</div>
						<div className="flex items-center space-x-4">
							<div className="flex items-center space-x-2">
								<div className="w-8 h-8 bg-[#FF8C00] rounded-full flex items-center justify-center">
									<span className="text-white text-sm">
										{user?.email?.[0].toUpperCase()}
									</span>
								</div>
								<span className="text-sm text-gray-600 hidden sm:inline">
									{user?.email}
								</span>
							</div>
							<button
								type="button"
								onClick={handleLogout}
								className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg"
							>
								Logout
							</button>
						</div>
					</div>
				</div>
			</div>
			
			<div className="flex-1 overflow-y-auto p-4">
				{error && (
					<div className="max-w-4xl mx-auto mb-4">
						<div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg">
							{error}
						</div>
					</div>
				)}
				
				{messages.length === 0 ? (
					<div className="flex items-center justify-center h-full text-gray-500">
						Start a conversation by typing a message below
					</div>
				) : (
					<div className="max-w-4xl mx-auto space-y-4">
						{messages.map((message) => (
							<div
								key={message.id}
								className={`flex ${
									message.role === 'user' ? 'justify-end' : 'justify-start'
								}`}
							>
								<div
									className={`max-w-[80%] rounded-lg p-4 ${
										message.role === 'user'
											? 'bg-[#FF8C00] text-white'
											: 'bg-white border shadow-sm'
									}`}
								>
									{message.content}
								</div>
							</div>
						))}
						{isLoading && (
							<div className="flex justify-start">
								<div className="max-w-[80%] rounded-lg p-4 bg-white border shadow-sm">
									<div className="flex items-center space-x-2">
										<div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
										<div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
										<div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
									</div>
								</div>
							</div>
						)}
						<div ref={messagesEndRef} />
					</div>
				)}
			</div>

			<div className="flex-none p-4 bg-white border-t">
				<div className="max-w-4xl mx-auto">
					<div className="flex space-x-2">
						<input
							type="text"
							value={input}
							onChange={(e) => setInput(e.target.value)}
							onKeyDown={handleKeyDown}
							placeholder="Type your message..."
							disabled={isLoading}
							className="flex-1 p-3 border rounded-lg focus:ring-2 focus:ring-[#FF8C00] focus:border-[#FF8C00] outline-none"
						/>
						<button
							type="button"
							onClick={handleSendMessage}
							disabled={isLoading}
							className="px-6 py-3 bg-[#FF8C00] text-white rounded-lg hover:bg-[#E67E00] disabled:opacity-50"
						>
							{isLoading ? 'Sending...' : 'Send'}
						</button>
					</div>
				</div>
			</div>
		</div>
	);
}