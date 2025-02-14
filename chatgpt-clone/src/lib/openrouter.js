import axios from 'axios';

const OPENROUTER_API_KEY = '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';
const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';

export const generateResponse = async (messages) => {
	try {
		const response = await axios.post(
			OPENROUTER_API_URL,
			{
				model: 'gpt-4',
				messages: messages,
			},
			{
				headers: {
					Authorization: `Bearer ${OPENROUTER_API_KEY}`,
					'Content-Type': 'application/json',
				},
			}
		);
		return response.data.choices[0].message.content;
	} catch (error) {
		console.error('Error calling OpenRouter API:', error);
		throw error;
	}
};