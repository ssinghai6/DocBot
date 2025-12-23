"use client";

import { useState } from 'react';
import FileUpload from '@/components/FileUpload';
import ChatInterface from '@/components/ChatInterface';
import { motion } from 'framer-motion';

import BackgroundAnimation from '@/components/BackgroundAnimation';

export default function Home() {
  const [documentText, setDocumentText] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  const handleUploadComplete = (text: string, fname: string) => {
    setDocumentText(text);
    setFilename(fname);
  };

  const handleClear = () => {
    setDocumentText(null);
    setFilename(null);
  };

  return (
    <main className="min-h-screen relative flex flex-col items-center justify-center p-4 overflow-hidden">
      {/* Background Ambience */}
      <BackgroundAnimation />

      <div className="z-10 w-full max-w-5xl">
        <motion.div
          layout
          transition={{ duration: 0.5, type: "spring", bounce: 0.2 }}
          className="w-full"
        >
          {!documentText ? (
            <div className="flex flex-col items-center space-y-12">
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
              >
                <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-white pb-2">
                  DocBot
                </h1>
                <p className="text-xl text-gray-400 max-w-lg mx-auto">
                  Your friendly PDF assistant. Powered by Gemini 1.5 Flash.
                </p>
              </motion.div>

              <FileUpload onUploadComplete={handleUploadComplete} />
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            >
              <ChatInterface
                documentText={documentText}
                filename={filename || 'Document.pdf'}
                onClear={handleClear}
              />
            </motion.div>
          )}
        </motion.div>
      </div>

      <div className="fixed bottom-4 text-center w-full text-xs text-gray-700 pointer-events-none">
        Developed by Sanshrit Singhai
      </div>
    </main>
  );
}
