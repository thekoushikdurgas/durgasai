import { NextResponse } from 'next/server';

export async function POST(req: Request) {
	try {
		const { message } = await req.json();
		
		const response = await fetch('https://api.openrouter.ai/api/v1/chat/completions', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
				'HTTP-Referer': process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
				'X-Title': 'Durgas AI Chat',
			},
			body: JSON.stringify({
				model: 'openai/gpt-3.5-turbo',
				messages: [{ role: 'user', content: message }],
				temperature: 0.7,
				max_tokens: 1000,
			}),
		});

		if (!response.ok) {
			throw new Error(`OpenRouter API error: ${response.status}`);
		}

		const data = await response.json();
		
		if (!data.choices?.[0]?.message?.content) {
			throw new Error('Invalid response format from OpenRouter API');
		}

		return NextResponse.json({
			message: data.choices[0].message.content,
		});
	} catch (error) {
		console.error('Error in chat API:', error);
		return NextResponse.json(
			{ error: 'Failed to process chat request' },
			{ status: 500 }
		);
	}
}