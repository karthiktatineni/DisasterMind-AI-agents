import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DisasterMind AI",
  description: "Multi-agent disaster intelligence and response planning dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

