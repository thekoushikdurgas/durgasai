import SignupForm from '@/components/auth/SignupForm';
import Head from 'next/head';

export default function Signup() {
	return (
		<>
			<Head>
				<title>Sign Up - Lucy™</title>
				<meta name="description" content="Join Lucy™ - Innovation Starts Here" />
			</Head>
			<SignupForm />
		</>
	);
}