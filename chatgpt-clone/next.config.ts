import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ['api.openrouter.ai'],
  },
  env: {
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
  },
};

export default nextConfig;

