"use client";

import { useState, useRef } from 'react';
import { Upload, FileText, Loader2, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface FileUploadProps {
  onUploadComplete: (text: string, filename: string) => void;
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const processFile = async (file: File) => {
    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file.');
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Assuming API is proxied or running on localhost:8000 for dev
      // In production, Next.js rewrites or Vercel functions handle this.
      // For local dev, we might need full URL if not using proxy.
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process PDF text');
      }

      const data = await response.json();
      onUploadComplete(data.text, data.filename);
    } catch (err: any) {
      setError(err.message || 'An error occurred during upload.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={twMerge(
          "relative group cursor-pointer border-2 border-dashed rounded-3xl p-10 transition-all duration-300 ease-out",
          isDragging ? "border-blue-500 bg-blue-500/10 scale-[1.02]" : "border-white/10 hover:border-white/20 hover:bg-white/5",
          "glass-card"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="application/pdf"
          onChange={handleFileSelect}
        />

        <div className="flex flex-col items-center justify-center text-center space-y-4">
          <div className={twMerge(
            "p-4 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 shadow-lg transition-transform duration-300",
            isDragging || isUploading ? "scale-110" : "group-hover:scale-110"
          )}>
            {isUploading ? (
              <Loader2 className="w-8 h-8 text-white animate-spin" />
            ) : (
              <Upload className="w-8 h-8 text-white" />
            )}
          </div>

          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-white">
              {isUploading ? "Processing Document..." : "Upload PDF"}
            </h3>
            <p className="text-gray-400 text-sm max-w-xs mx-auto">
              {isUploading
                ? "Extracting text relative to AI context window..."
                : "Drag & drop your PDF here, or click to browse"}
            </p>
          </div>
        </div>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-200 text-sm text-center"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
